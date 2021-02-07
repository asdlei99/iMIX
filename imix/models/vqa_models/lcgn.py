from ..builder import VQA_MODELS, build_backbone, build_embedding, build_encoder, build_head, build_combine_layer
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from imix.models.backbones.lcgn_backbone import Linear, apply_mask1d
from .base_model import BaseModel


@VQA_MODELS.register_module()
class LCGN(BaseModel):

  def __init__(self, encoder, backbone, head):
    super().__init__()

    self.encoder_model = build_encoder(encoder)
    self.backbone = build_backbone(backbone)
    self.single_hop = SingleHop(self.backbone.CTX_DIM,
                                self.encoder_model.ENC_DIM)
    self.head = build_head(head)  ###包括 classification head， generation head

  def forward_train(self, data):
    questionIndices = data['input_ids'].cuda()
    questionLengths = data['questionLengths']
    images = torch.cat((data['feature'], data['bbox'].repeat(1, 1, 16)),
                       dim=-1).cuda()
    batchSize = data['image_dim'].shape[0]
    imagesObjectNum = data['image_dim'].cuda()

    questionCntxWords, vecQuestions = self.encoder_model(
        questionIndices, questionLengths)

    # LCGN
    x_out = self.backbone(
        images=images,
        q_encoding=vecQuestions,
        lstm_outputs=questionCntxWords,
        q_length=questionLengths,
        entity_num=imagesObjectNum)

    # Single-Hop
    x_att = self.single_hop(x_out, vecQuestions, imagesObjectNum)
    model_output = {
        'scores': self.head.forward(x_att, vecQuestions),
        'target': data['answers_scores'].cuda()
    }
    return model_output

    # loss = self.head.forward_train(
    #     x_att, vecQuestions, labels=data['answers_scores'].cuda())
    #
    # return {'losses': loss}


class SingleHop(nn.Module):

  def __init__(self, CTX_DIM, ENC_DIM):
    super().__init__()
    self.CTX_DIM = CTX_DIM
    self.ENC_DIM = ENC_DIM

    self.proj_q = Linear(self.ENC_DIM, self.CTX_DIM)
    self.inter2att = Linear(self.CTX_DIM, 1)

  def forward(self, kb, vecQuestions, imagesObjectNum):
    proj_q = self.proj_q(vecQuestions)
    interactions = F.normalize(kb * proj_q[:, None, :], dim=-1)
    raw_att = self.inter2att(interactions).squeeze(-1)
    raw_att = apply_mask1d(raw_att, imagesObjectNum)
    att = F.softmax(raw_att, dim=-1)

    x_att = torch.bmm(att[:, None, :], kb).squeeze(1)
    return x_att
