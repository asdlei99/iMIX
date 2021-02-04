from ..builder import VQA_MODELS, build_backbone, build_embedding, build_encoder, build_head, build_combine_layer
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from mix.models.backbones.lcgn_backbone import Linear, apply_mask1d
from .base_model import BaseModel


@VQA_MODELS.register_module()
class BAN(BaseModel):

  def __init__(self, embedding, backbone, head):
    super().__init__()

    self.embeddimg_model = build_embedding(embedding)
    # self.encoder_model = build_encoder(encoder)
    self.backbone = build_backbone(backbone)
    # self.combine_model = build_combine_layer(combine_model)
    self.head = build_head(head)  ###包括 classification head， generation head

  def forward_train(self, data):
    v = data['feature'].cuda()
    q = self.embeddimg_model[0](data['input_ids'].cuda())
    q_emb = self.embeddimg_model[1].forward_all(q)
    q_emb = self.backbone(v, q_emb)
    targets = data['answers_scores'].cuda()
    logits = self.head.forward_train(q_emb.sum(1), labels=targets)

    return {'loss': logits}
