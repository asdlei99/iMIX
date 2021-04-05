from ..builder import VQA_MODELS, build_backbone, build_embedding, build_encoder, build_head, build_combine_layer
import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    # BertLayerNorm,
    BertPreTrainedModel,
)
from .base_model import BaseModel
from ..encoder import ViLBERTForPretraining, ViLBERTForClassification


@VQA_MODELS.register_module()
class VilBERT(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()

        params = kwargs['params']
        # self.special_visual_initialize = params['special_visual_initialize']
        freeze_base = params['freeze_base']
        self.training_head_type = params['training_head_type']
        if self.training_head_type == 'pretraining':
            self.model = ViLBERTForPretraining(**params)
        else:
            self.model = ViLBERTForClassification(**params)

        # if self.special_visual_initialize:
        #     self.model.bert.embeddings.initialize_visual_from_pretrained()

        if freeze_base:
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def get_image_and_text_features(self, data):
        bert_input_ids = data['input_ids']
        bert_input_mask = data['input_mask']
        bert_input_type_ids = data['input_segment']

        # if sample_list.dataset_name == 'nlvr2':
        #     bert_input_ids = torch.cat([bert_input_ids, bert_input_ids])
        #     bert_input_mask = torch.cat([bert_input_mask, bert_input_mask])
        #     bert_input_type_ids = torch.cat(
        #         [bert_input_type_ids, bert_input_type_ids])
        #
        #     # image input
        #     img0 = getattr(sample_list, 'img0', {})
        #     image_info = getattr(img0, 'image_info_0', {})
        #     image_dim_variable_0 = getattr(image_info, 'max_features', None)
        #     image_feature_variable_0 = getattr(img0, 'image_feature_0', None)
        #     image_location_variable_0 = getattr(image_info, 'bbox', None)
        #
        #     img1 = getattr(sample_list, 'img1', {})
        #     image_info = getattr(img1, 'image_info_0', {})
        #     image_dim_variable_1 = getattr(image_info, 'max_features', None)
        #     image_feature_variable_1 = getattr(img1, 'image_feature_0', None)
        #     image_location_variable_1 = getattr(image_info, 'bbox', None)
        #
        #     image_feature_variable = torch.cat(
        #         [image_feature_variable_0, image_feature_variable_1])
        #     image_location_variable = torch.cat(
        #         [image_location_variable_0, image_location_variable_1])
        #     image_dim_variable = torch.cat(
        #         [image_dim_variable_0, image_dim_variable_1])
        #     image_label_variable = None
        #     image_target_variable = None
        # else:
        # image_info = getattr(sample_list, 'image_info_0', {})
        image_dim_variable = data['image_dim']
        image_feature_variable = data['feature']
        image_label_variable = None
        image_location_variable = torch.cat([data['bbox'][:, :, :4], data['bbox'][:, :, -1].unsqueeze(-1)], dim=-1)
        image_target_variable = data['cls_prob']
        answers = data['answers_scores']

        return {
            'input_ids': bert_input_ids,
            'attention_mask': bert_input_mask,
            'token_type_ids': bert_input_type_ids,
            'image_dim': image_dim_variable,
            'image_feature': image_feature_variable,
            'image_location': image_location_variable,
            'image_target': image_target_variable,
            'image_label': image_label_variable,
            'ans': answers
        }

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def forward_train(self, data):
        params = self.get_image_and_text_features(data)
        params['masked_lm_labels'] = data['input_lm_label_ids']
        # Prepare Mask
        if params['image_feature'] is not None and params['image_dim'] is not None:
            image_mask = (torch.arange(params['image_feature'].size(-2)).expand(*params['image_feature'].size()[:-1]))
            if len(params['image_dim'].size()) < len(image_mask.size()):
                params['image_dim'] = data['image_dim'].unsqueeze(-1)
                assert len(params['image_dim'].size()) == len(image_mask.size())
            image_mask = image_mask < params['image_dim']
            params['image_attention_mask'] = image_mask.long()
        else:
            params['image_attention_mask'] = None
        output_dict = self.model(
            params['input_ids'].cuda(),
            params['image_feature'].cuda(),
            params['image_location'].cuda(),
            params['token_type_ids'].cuda(),
            params['attention_mask'].cuda(),
            params['image_attention_mask'].cuda(),
            params['masked_lm_labels'].cuda(),
            None,
            params['image_target'].cuda(),
        )

        model_output = {'scores': output_dict['scores'], 'target': params['ans'].cuda()}
        return model_output
        # losses = F.binary_cross_entropy_with_logits(output_dict['scores'],
        #                                             params['ans'].cuda())
        # return {'losses': losses}
