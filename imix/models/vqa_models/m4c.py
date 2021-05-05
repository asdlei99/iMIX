import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import VQA_MODELS, build_backbone, build_encoder, build_head
from .base_model import BaseModel


# test forward####
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


#################


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@VQA_MODELS.register_module()
class M4C(BaseModel):

    def __init__(self, hidden_dim, dropout_prob, ocr_in_dim, encoder, backbone, combine_model, head):
        super().__init__()

        self.encoder_model = build_encoder(encoder)
        self.backbone = build_backbone(backbone)
        # self.combine_model = build_combine_layer(combine_model)  ###combine text and image
        self.head = build_head(head)
        # self.loss = M4CDecodingBCEWithMaskLoss()

        self.finetune_modules = []

        self.finetune_modules.append({
            'module': self.encoder_model[0].text_bert,
            'lr_scale': 0.1
        })  # self.config.lr_scale_text_bert

        self.finetune_modules.append({'module': self.encoder_model[1], 'lr_scale': 0.1})  # self.config.lr_scale_frcn

        self.finetune_modules.append({'module': self.encoder_model[2], 'lr_scale': 0.1})  # self.config.lr_scale_frcn

        # self.init_weights()
        # apply smaller lr to pretrained Faster R-CNN fc7
        # self.finetune_modules.append(
        #     {"module": self.obj_faster_rcnn_fc7, "lr_scale": self.config.lr_scale_frcn}
        # )

        self.linear_obj_feat_to_mmt_in = nn.Linear(self.encoder_model[1].out_dim, hidden_dim)

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, hidden_dim)

        self.obj_feat_layer_norm = nn.LayerNorm(hidden_dim)
        self.obj_bbox_layer_norm = nn.LayerNorm(hidden_dim)
        self.obj_drop = nn.Dropout(dropout_prob)

        # self.finetune_modules.append(
        #     {"module": self.ocr_faster_rcnn_fc7, "lr_scale": self.config.lr_scale_frcn}
        # )

        self.linear_ocr_feat_to_mmt_in = nn.Linear(ocr_in_dim, hidden_dim)

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, hidden_dim)

        self.ocr_feat_layer_norm = nn.LayerNorm(hidden_dim)
        self.ocr_bbox_layer_norm = nn.LayerNorm(hidden_dim)
        self.ocr_drop = nn.Dropout(dropout_prob)

        self.ocr_ptr_net = OcrPtrNet(hidden_dim, hidden_dim)

        # self.init_weights()

    def _forward_txt_encoding(self, data, fwd_results):
        fwd_results['txt_inds'] = data['input_ids'].cuda()
        # text_len = data['input_ids'].shape[1] * torch.ones(data['input_ids'].shape[0])
        # binary mask of valid text (question words) vs padding
        # fwd_results['txt_mask'] = _get_mask(text_len, data['input_ids'].size(1)).cuda()
        fwd_results['txt_mask'] = data['input_mask'].cuda()
        fwd_results['prev_inds'] = data['train_prev_inds'].cuda()
        fwd_results['answers_scores'] = data['answers_scores'].cuda()
        fwd_results['train_loss_mask'] = data['train_loss_mask'].cuda()
        # text_bert_emb = self.embedding_model.word_embedding(fwd_results["txt_inds"])
        # text_bert_out = self.backbone.text_encoder(txt_inds=fwd_results["txt_inds"], txt_mask=fwd_results["txt_mask"])
        # fwd_results["txt_emb"] = self.backbone.text_bert_out_linear(text_bert_out)

    def _forward_obj_encoding(self, data, fwd_results):
        # object appearance feature: Faster R-CNN fc7
        obj_fc6 = data['feature'].cuda()
        obj_fc7 = self.encoder_model[1](obj_fc6)
        obj_fc7 = F.normalize(obj_fc7, dim=-1)

        obj_feat = obj_fc7
        obj_bbox = data['bbox_normalized'].cuda()
        obj_mmt_in = self.obj_feat_layer_norm(self.linear_obj_feat_to_mmt_in(obj_feat)) + self.obj_bbox_layer_norm(
            self.linear_obj_bbox_to_mmt_in(obj_bbox))
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results['obj_mmt_in'] = obj_mmt_in

        # binary mask of valid object vs padding
        obj_nums = obj_fc6.shape[1] * torch.ones(obj_fc6.shape[0])
        fwd_results['obj_mask'] = _get_mask(obj_nums, obj_mmt_in.size(1)).cuda()

    def _forward_ocr_encoding(self, data, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = data['ocr_vectors_fasttext'].cuda()
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = data['ocr_vectors_phoc'].cuda()
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # OCR appearance feature: Faster R-CNN fc7
        ocr_fc6 = data['feature_ocr'].cuda()
        ocr_fc7 = self.encoder_model[-1](ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)

        # OCR order vectors (legacy from LoRRA model; set to all zeros)
        # TODO remove OCR order vectors; they are not needed
        ocr_order_vectors = torch.zeros_like(data['ocr_vectors_order'].cuda())

        # if self.remove_ocr_fasttext:
        #     ocr_fasttext = torch.zeros_like(ocr_fasttext)
        # if self.remove_ocr_phoc:
        #     ocr_phoc = torch.zeros_like(ocr_phoc)
        # if self.remove_ocr_frcn:
        #     ocr_fc7 = torch.zeros_like(ocr_fc7)
        ocr_feat = torch.cat([ocr_fasttext, ocr_phoc, ocr_fc7, ocr_order_vectors], dim=-1)
        ocr_bbox = data['bbox_ocr_normalized'].cuda()
        # if self.remove_ocr_semantics:
        #     ocr_feat = torch.zeros_like(ocr_feat)
        # if self.remove_ocr_bbox:
        #     ocr_bbox = torch.zeros_like(ocr_bbox)
        ocr_mmt_in = self.ocr_feat_layer_norm(self.linear_ocr_feat_to_mmt_in(ocr_feat)) + self.ocr_bbox_layer_norm(
            self.linear_ocr_bbox_to_mmt_in(ocr_bbox))
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results['ocr_mmt_in'] = ocr_mmt_in

        # binary mask of valid OCR vs padding
        ocr_nums = ocr_fc6.shape[1] * torch.ones(ocr_fc6.shape[0])
        fwd_results['ocr_mask'] = _get_mask(ocr_nums, ocr_mmt_in.size(1)).cuda()

    def _forward_output(self, fwd_results):
        mmt_dec_output = fwd_results['mmt_dec_output']
        mmt_ocr_output = fwd_results['mmt_ocr_output']
        ocr_mask = fwd_results['ocr_mask']

        fixed_scores = self.head(mmt_dec_output)
        dynamic_ocr_scores = self.ocr_ptr_net(mmt_dec_output, mmt_ocr_output, ocr_mask)
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
        fwd_results['scores'] = scores

    def forward_train(self, data, *args, **kwargs):
        fwd_results = {}
        self._forward_txt_encoding(data, fwd_results)
        self._forward_obj_encoding(data, fwd_results)
        self._forward_ocr_encoding(data, fwd_results)

        text_bert_out = self.encoder_model[0](txt_inds=fwd_results['txt_inds'], txt_mask=fwd_results['txt_mask'])
        fwd_results['txt_emb'] = text_bert_out
        mmt_results = self.backbone(
            txt_emb=fwd_results['txt_emb'],
            txt_mask=fwd_results['txt_mask'],
            obj_emb=fwd_results['obj_mmt_in'],
            obj_mask=fwd_results['obj_mask'],
            ocr_emb=fwd_results['ocr_mmt_in'],
            ocr_mask=fwd_results['ocr_mask'],
            fixed_ans_emb=self.head.module.weight,
            prev_inds=fwd_results['prev_inds'])

        fwd_results.update(mmt_results)
        self._forward_output(fwd_results)
        scores = fwd_results['scores']
        train_loss_mask = fwd_results['train_loss_mask']
        targets = fwd_results['answers_scores']
        # loss = self.loss(scores, train_loss_mask, targets)
        model_outputs = {'scores': scores, 'target': targets, 'train_loss_mask': train_loss_mask}

        return model_outputs

    def forward_test(self, data, *args, **kwargs):
        model_output = self.forward_train(data)
        return model_output


class OcrPtrNet(nn.Module):

    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        assert extended_attention_mask.dim() == 2
        extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores
