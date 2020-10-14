from ..builder import VQA_MODELS, build_backbone, build_embedding, build_encoder, build_head, build_combine_layer
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertLayerNorm,
    BertPreTrainedModel,
)

from mix.models.encoder import ViLBERTForPretraining, ViLBERTForClassification


@VQA_MODELS.register_module()
class VilBERT(nn.Module):

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

    def get_image_and_text_features(self, sample_list):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids

        if sample_list.dataset_name == 'nlvr2':
            bert_input_ids = torch.cat([bert_input_ids, bert_input_ids])
            bert_input_mask = torch.cat([bert_input_mask, bert_input_mask])
            bert_input_type_ids = torch.cat(
                [bert_input_type_ids, bert_input_type_ids])

            # image input
            img0 = getattr(sample_list, 'img0', {})
            image_info = getattr(img0, 'image_info_0', {})
            image_dim_variable_0 = getattr(image_info, 'max_features', None)
            image_feature_variable_0 = getattr(img0, 'image_feature_0', None)
            image_location_variable_0 = getattr(image_info, 'bbox', None)

            img1 = getattr(sample_list, 'img1', {})
            image_info = getattr(img1, 'image_info_0', {})
            image_dim_variable_1 = getattr(image_info, 'max_features', None)
            image_feature_variable_1 = getattr(img1, 'image_feature_0', None)
            image_location_variable_1 = getattr(image_info, 'bbox', None)

            image_feature_variable = torch.cat(
                [image_feature_variable_0, image_feature_variable_1])
            image_location_variable = torch.cat(
                [image_location_variable_0, image_location_variable_1])
            image_dim_variable = torch.cat(
                [image_dim_variable_0, image_dim_variable_1])
            image_label_variable = None
            image_target_variable = None
        else:
            image_info = getattr(sample_list, 'image_info_0', {})
            image_dim_variable = getattr(image_info, 'max_features', None)
            image_feature_variable = getattr(sample_list, 'image_feature_0',
                                             None)
            image_label_variable = getattr(sample_list, 'image_labels', None)
            image_location_variable = getattr(image_info, 'bbox', None)

            cls_prob = getattr(image_info, 'cls_prob', None)
            image_target = np.array(cls_prob, dtype=np.float32)
            image_target_variable = torch.tensor(
                image_target, dtype=torch.float).cuda()

        return {
            'input_ids': bert_input_ids,
            'attention_mask': bert_input_mask,
            'token_type_ids': bert_input_type_ids,
            'image_dim': image_dim_variable,
            'image_feature': image_feature_variable,
            'image_location': image_location_variable,
            'image_target': image_target_variable,
            'image_label': image_label_variable,
        }

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def forward(self, data):
        # sample_list = self.update_sample_list_based_on_head(sample_list)
        # sample_list = self.add_custom_params(sample_list)
        # sample_list = self.flatten_for_bert(sample_list)

        output_dict = self.model(
            data['input_ids'],
            data['image_feature'],
            data['image_location'],
            data['token_type_ids'],
            data['attention_mask'],
            data['image_attention_mask'],
            data['masked_lm_labels'],
            data['image_label'],
            data['image_target'],
        )

        return output_dict
