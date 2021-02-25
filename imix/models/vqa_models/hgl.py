from ..builder import VQA_MODELS, build_backbone, build_embedding, build_encoder, build_head, build_combine_layer
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import allennlp
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
import torchvision
from mmcv.ops import RoIAlign  ##TODO need to squeeze from mmdet(zhangrunze)
from torchvision.models import resnet
from .base_model import BaseModel
from .r2c import R2C
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    # BertLayerNorm,
    BertPreTrainedModel,
)


@VQA_MODELS.register_module()
class HGL(R2C):

  def __init__(self, input_dropout, pretrained, average_pool, semantic,
               final_dim, backbone, head):
    super(HGL, self).__init__(input_dropout, pretrained, average_pool, semantic,
                              final_dim, backbone, head)

  def forward_train(self, data):
    # images = data['images']
    # boxes = data['boxes']
    # box_mask = data['box_mask']
    # objects = data['objects']
    # segms = data['segms']
    # question = data['question']
    # question_tags = data['question_tags']
    # question_mask = data['question_mask']
    # answers = data['answers']
    # answer_tags = data['answer_tags']
    # answer_mask = data['answer_mask']

    images = data['image'].cuda()
    max_num = torch.max(data['max_num']).cuda()
    max_bbox_num = torch.max(data['bbox_num']).cuda()
    boxes = data['boxes'][:, :max_bbox_num, :].cuda()
    # box_mask = data['box_mask'][:, :max_bbox_num, :]
    bbox_mask = torch.all(boxes >= 0, -1).long().cuda()
    objects = data['objects'][:, :max_bbox_num].cuda()
    segms = data['segms'][:, :max_bbox_num, :, :].cuda()
    question = data['questions_embeddings'][:, :, :max_num, :].cuda()
    question_tags = data['questions_obj_tags'][:, :, :max_num].cuda()
    question_mask = data['questions_masks'][:, :, :max_num].cuda()
    answers = data['answers_embeddings'][:, :, :max_num, :].cuda()
    answer_tags = data['answers_obj_tags'][:, :, :max_num].cuda()
    answer_mask = data['answers_masks'][:, :, :max_num].cuda()

    obj_reps = self.detector(
        images=images,
        boxes=boxes,
        box_mask=bbox_mask,
        classes=objects,
        segms=segms)
    q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask,
                                        obj_reps['obj_reps'])
    a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask,
                                        obj_reps['obj_reps'])

    pooled_rep = self.backbone(bbox_mask, a_rep, q_rep, obj_reps, question_mask,
                               answer_mask)
    logits = self.head(pooled_rep)

    model_output = {'scores': logits.squeeze(2),
                    'target': data['label'].cuda(),
                    'obj_scores': obj_reps['obj_logits'],
                    'obj_target':  obj_reps['obj_labels']}


    return model_output

  def forward_test(self, data):
    model_output = self.forward_train(data)
    return model_output