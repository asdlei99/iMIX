from ..builder import VQA_MODELS, build_backbone, build_embedding, build_encoder, build_head, build_combine_layer
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from imix.models.backbones.lcgn_backbone import Linear, apply_mask1d
from .base_model import BaseModel
from pytorch_pretrained_bert.modeling import BertModel
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


# def yolo_loss(input,
#               target,
#               gi,
#               gj,
#               best_n_list,
#               w_coord=5.,
#               w_neg=1. / 5,
#               size_average=True):
#   mseloss = torch.nn.MSELoss(size_average=True)
#   celoss = torch.nn.CrossEntropyLoss(size_average=True)
#   batch = input.size(0)
#
#   pred_bbox = Variable(torch.zeros(batch, 4).cuda())
#   gt_bbox = Variable(torch.zeros(batch, 4).cuda())
#   for ii in range(batch):
#     pred_bbox[ii, 0:2] = F.sigmoid(input[ii, best_n_list[ii], 0:2, gj[ii],
#                                          gi[ii]])
#     pred_bbox[ii, 2:4] = input[ii, best_n_list[ii], 2:4, gj[ii], gi[ii]]
#     gt_bbox[ii, :] = target[ii, best_n_list[ii], :4, gj[ii], gi[ii]]
#   loss_x = mseloss(pred_bbox[:, 0], gt_bbox[:, 0])
#   loss_y = mseloss(pred_bbox[:, 1], gt_bbox[:, 1])
#   loss_w = mseloss(pred_bbox[:, 2], gt_bbox[:, 2])
#   loss_h = mseloss(pred_bbox[:, 3], gt_bbox[:, 3])
#
#   pred_conf_list, gt_conf_list = [], []
#   pred_conf_list.append(input[:, :, 4, :, :].contiguous().view(batch, -1))
#   gt_conf_list.append(target[:, :, 4, :, :].contiguous().view(batch, -1))
#   pred_conf = torch.cat(pred_conf_list, dim=1)
#   gt_conf = torch.cat(gt_conf_list, dim=1)
#   loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
#   return (loss_x + loss_y + loss_w + loss_h) * w_coord + loss_conf
#
#
# def bbox_iou(box1, box2, x1y1x2y2=True):
#   """Returns the IoU of two bounding boxes."""
#   if x1y1x2y2:
#     # Get the coordinates of bounding boxes
#     b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
#     b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
#   else:
#     # Transform from center and width to exact coordinates
#     b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
#     b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
#     b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
#     b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
#
#   # get the coordinates of the intersection rectangle
#   inter_rect_x1 = torch.max(b1_x1, b2_x1)
#   inter_rect_y1 = torch.max(b1_y1, b2_y1)
#   inter_rect_x2 = torch.min(b1_x2, b2_x2)
#   inter_rect_y2 = torch.min(b1_y2, b2_y2)
#   # Intersection area
#   inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(
#       inter_rect_y2 - inter_rect_y1, 0)
#   # Union Area
#   b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
#   b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
#
#   # print(box1, box1.shape)
#   # print(box2, box2.shape)
#   return inter_area / (b1_area + b2_area - inter_area + 1e-16)
#
#
# def mask_softmax(attn_score,
#                  word_mask,
#                  tempuature=10.,
#                  clssep=False,
#                  lstm=False):
#   if len(attn_score.shape) != 2:
#     attn_score = attn_score.squeeze(2).squeeze(2)
#   word_mask_cp = word_mask[:, :attn_score.shape[1]].clone()
#   score = F.softmax(attn_score * tempuature, dim=1)
#   if not clssep:
#     for ii in range(word_mask_cp.shape[0]):
#       if lstm:
#         word_mask_cp[ii, word_mask_cp[ii, :].sum() - 1] = 0
#       else:
#         word_mask_cp[ii, 0] = 0
#         word_mask_cp[ii, word_mask_cp[ii, :].sum()] = 0  ## set one to 0 already
#   mask_score = score * word_mask_cp.float()
#   mask_score = mask_score / (mask_score.sum(1) + 1e-8).view(
#       mask_score.size(0), 1).expand(mask_score.size(0), mask_score.size(1))
#   return mask_score
#
#
# def diverse_loss(score_list, word_mask, m=-1, coverage_reg=True):
#   score_matrix = torch.stack(
#       [mask_softmax(score, word_mask) for score in score_list],
#       dim=1)  ## (B,Nfilm,N,H,W)
#   cov_matrix = torch.bmm(score_matrix,
#                          score_matrix.permute(0, 2, 1))  ## (BHW,Nfilm,Nfilm)
#   id_matrix = Variable(
#       torch.eye(cov_matrix.shape[1]).unsqueeze(0).repeat(
#           cov_matrix.shape[0], 1, 1).cuda())
#   if m == -1.:
#     div_reg = torch.sum(
#         ((cov_matrix * (1 - id_matrix))**2).view(-1)) / cov_matrix.shape[0]
#   else:
#     div_reg = torch.sum(
#         ((cov_matrix - m * id_matrix)**2).view(-1)) / cov_matrix.shape[0]
#   if coverage_reg:
#     word_mask_cp = word_mask.clone()
#     for ii in range(word_mask_cp.shape[0]):
#       word_mask_cp[ii, 0] = 0
#       word_mask_cp[ii, word_mask_cp[ii, :].sum()] = 0  ## set one to 0 already
#     cover_matrix = 1. - torch.clamp(
#         torch.sum(score_matrix, dim=1, keepdim=False), min=0., max=1.)
#     cover_reg = torch.sum(
#         (cover_matrix * word_mask_cp.float()).view(-1)) / cov_matrix.shape[0]
#     div_reg += cover_reg
#   return div_reg
#
#
# def build_target(raw_coord, pred, anchors_full):
#   size = 256
#   gsize = 8
#   coord = Variable(torch.zeros(raw_coord.size(0), raw_coord.size(1)).cuda())
#   batch, grid = raw_coord.size(0), size // gsize
#   coord[:, 0] = (raw_coord[:, 0] + raw_coord[:, 2]) / (2 * size)
#   coord[:, 1] = (raw_coord[:, 1] + raw_coord[:, 3]) / (2 * size)
#   coord[:, 2] = (raw_coord[:, 2] - raw_coord[:, 0]) / size
#   coord[:, 3] = (raw_coord[:, 3] - raw_coord[:, 1]) / size
#   coord = coord * grid
#   bbox = torch.zeros(coord.size(0), 9, 5, grid, grid)
#
#   best_n_list, best_gi, best_gj = [], [], []
#
#   for ii in range(batch):
#     batch, grid = raw_coord.size(0), size // gsize
#     gi = coord[ii, 0].long()
#     gj = coord[ii, 1].long()
#     tx = coord[ii, 0] - gi.float()
#     ty = coord[ii, 1] - gj.float()
#     gw = coord[ii, 2]
#     gh = coord[ii, 3]
#
#     anchor_idxs = range(9)
#     anchor_imsize = 416
#     anchors = [anchors_full[i] for i in anchor_idxs]
#     scaled_anchors = [(x[0] / (anchor_imsize / grid), \
#                        x[1] / (anchor_imsize / grid)) for x in anchors]
#
#     ## Get shape of gt box
#     gt_box = torch.FloatTensor(np.array([0, 0, gw, gh],
#                                         dtype=np.float32)).unsqueeze(0)
#     ## Get shape of anchor box
#     anchor_shapes = torch.FloatTensor(
#         np.concatenate((np.zeros(
#             (len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))
#     ## Calculate iou between gt and anchor shapes
#     # anch_ious = list(bbox_iou(gt_box, anchor_shapes))
#     anch_ious = list(bbox_iou(gt_box, anchor_shapes, x1y1x2y2=False))
#     ## Find the best matching anchor box
#     best_n = np.argmax(np.array(anch_ious))
#
#     tw = torch.log(gw / scaled_anchors[best_n][0] + 1e-16)
#     th = torch.log(gh / scaled_anchors[best_n][1] + 1e-16)
#
#     bbox[ii, best_n, :, gj,
#          gi] = torch.stack([tx, ty, tw, th,
#                             torch.ones(1).cuda().squeeze()])
#     best_n_list.append(int(best_n))
#     best_gi.append(gi)
#     best_gj.append(gj)
#
#   bbox = Variable(bbox.cuda())
#   return bbox, best_gi, best_gj, best_n_list
#
#
# @VQA_MODELS.register_module()
# class ReSC(BaseModel):
#
#   def __init__(self, encoder, backbone, weights_file):
#     super().__init__()
#
#     self.encoder_model = build_encoder(encoder)
#     self.encoder_model.load_weights(weights_file)  ###TODO zhangrunze
#     self.textmodel = BertModel.from_pretrained('bert-base-uncased')
#     self.backbone = build_backbone(backbone)
#     anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
#     anchors = [float(x) for x in anchors.split(',')]
#     self.anchors_full = [
#         (anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)
#     ][::-1]
#     # self.head = build_head(
#     #     head)  ###包括 classification head， generation head
#
#   def compute_loss(self, pred_anchor_list, bbox, anchors_full, attnscore_list,
#                    word_mask):
#     loss = 0.
#     for pred_anchor in pred_anchor_list:
#       ## convert gt box to center+offset format
#       gt_param, gi, gj, best_n_list = build_target(bbox, pred_anchor,
#                                                    anchors_full)
#       ## flatten anchor dim at each scale
#       pred_anchor = pred_anchor.view( \
#           pred_anchor.size(0), 9, 5, pred_anchor.size(2), pred_anchor.size(3))
#       ## loss
#       loss += yolo_loss(pred_anchor, gt_param, gi, gj, best_n_list)
#     losses = {'loss': loss}
#
#     pred_anchor = pred_anchor_list[-1].view(pred_anchor_list[-1].size(0), \
#                                             9, 5, pred_anchor_list[-1].size(2), pred_anchor_list[-1].size(3))
#
#     ## diversity regularization
#     w_div = 0.125
#     div_loss = diverse_loss(attnscore_list, word_mask) * w_div
#     losses['div_loss'] = div_loss
#     return losses
#
#   def forward_train(self, data):
#     input_mask = data['input_mask'].cuda()
#     image = data['image'].cuda()
#     input_ids = data['input_ids'].cuda()
#
#     raw_fvisu = self.encoder_model(image)
#     raw_fvisu = raw_fvisu[2]
#
#     ## Language Module
#     all_encoder_layers, _ = self.textmodel(
#         input_ids, token_type_ids=None, attention_mask=input_mask)
#     ## Sentence feature at the first position [cls]
#     raw_flang = (all_encoder_layers[-1][:, 0, :] + all_encoder_layers[-2][:, 0, :] \
#                  + all_encoder_layers[-3][:, 0, :] + all_encoder_layers[-4][:, 0, :]) / 4
#     raw_fword = (all_encoder_layers[-1] + all_encoder_layers[-2] \
#                  + all_encoder_layers[-3] + all_encoder_layers[-4]) / 4
#     raw_flang = raw_flang.detach()
#     raw_fword = raw_fword.detach()
#
#     pred_anchor_list, attnscore_list = self.backbone(raw_fword, raw_fvisu,
#                                                      input_mask)
#     bbox = data['bbox'].cuda()
#     anchors_full = self.anchors_full
#
#     losses = self.compute_loss(pred_anchor_list, bbox, anchors_full,
#                                attnscore_list, input_mask)
#     return losses


@VQA_MODELS.register_module()
class ReSC(BaseModel):

  def __init__(self, encoder, backbone, weights_file):
    super().__init__()

    self.encoder_model = build_encoder(encoder)
    self.encoder_model.load_weights(weights_file)  ###TODO zhangrunze
    self.textmodel = BertModel.from_pretrained('bert-base-uncased')
    self.backbone = build_backbone(backbone)
    # anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
    # anchors = [float(x) for x in anchors.split(',')]
    # self.anchors_full = [
    #     (anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)
    # ][::-1]
    # self.head = build_head(
    #     head)  ###包括 classification head， generation head

  # def compute_loss(self, pred_anchor_list, bbox, anchors_full, attnscore_list,
  #                  word_mask):
  #   loss = 0.
  #   for pred_anchor in pred_anchor_list:
  #     ## convert gt box to center+offset format
  #     gt_param, gi, gj, best_n_list = build_target(bbox, pred_anchor,
  #                                                  anchors_full)
  #     ## flatten anchor dim at each scale
  #     pred_anchor = pred_anchor.view( \
  #         pred_anchor.size(0), 9, 5, pred_anchor.size(2), pred_anchor.size(3))
  #     ## loss
  #     loss += yolo_loss(pred_anchor, gt_param, gi, gj, best_n_list)
  #   losses = {'loss': loss}
  #
  #   pred_anchor = pred_anchor_list[-1].view(pred_anchor_list[-1].size(0), \
  #                                           9, 5, pred_anchor_list[-1].size(2), pred_anchor_list[-1].size(3))
  #
  #   ## diversity regularization
  #   w_div = 0.125
  #   div_loss = diverse_loss(attnscore_list, word_mask) * w_div
  #   losses['div_loss'] = div_loss
  #   return losses

  def forward_train(self, data):
    input_mask = data['input_mask'].cuda()
    image = data['image'].cuda()
    input_ids = data['input_ids'].cuda()

    raw_fvisu = self.encoder_model(image)
    raw_fvisu = raw_fvisu[2]

    ## Language Module
    all_encoder_layers, _ = self.textmodel(
        input_ids, token_type_ids=None, attention_mask=input_mask)
    ## Sentence feature at the first position [cls]
    raw_flang = (all_encoder_layers[-1][:, 0, :] + all_encoder_layers[-2][:, 0, :] \
                 + all_encoder_layers[-3][:, 0, :] + all_encoder_layers[-4][:, 0, :]) / 4
    raw_fword = (all_encoder_layers[-1] + all_encoder_layers[-2] \
                 + all_encoder_layers[-3] + all_encoder_layers[-4]) / 4
    raw_flang = raw_flang.detach()
    raw_fword = raw_fword.detach()

    pred_anchor_list, attnscore_list = self.backbone(raw_fword, raw_fvisu,
                                                     input_mask)
    bbox = data['bbox'].cuda()
    # anchors_full = self.anchors_full

    # losses = self.compute_loss(pred_anchor_list, bbox, anchors_full,
    #                            attnscore_list, input_mask)

    model_output = {'scores': pred_anchor_list, 'target':bbox, 'input_mask': input_mask, "attnscore_list": attnscore_list}

    return model_output
