from ..builder import VQA_MODELS, build_backbone, build_embedding, build_encoder, build_head, build_combine_layer
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    # BertLayerNorm,
    BertPreTrainedModel,
)

from ..encoder import LXMERTForPretraining, LXMERTForClassification
from .base_model import BaseModel


@VQA_MODELS.register_module()
class LXMERT(BaseModel):

  def __init__(self, **kwargs):
    super().__init__()

    params = kwargs['params']
    # self.special_visual_initialize = params['special_visual_initialize']
    freeze_base = params['freeze_base']
    self.training_head_type = params['training_head_type']

    if self.training_head_type == 'pretraining':
      self.model = LXMERTForPretraining(**params)
    else:
      self.model = LXMERTForClassification(**params)

    # if self.special_visual_initialize:
    #     self.model.bert.embeddings.initialize_visual_from_pretrained()

    if freeze_base:
      for p in self.model.bert.parameters():
        p.requires_grad = False

  def get_image_and_text_features(self, data):
    # bert input
    bert_input_ids = data['input_ids']
    bert_input_mask = data['input_mask']
    bert_input_type_ids = data['input_segment']
    masked_lm_labels = data['input_lm_label_ids']

    # image input
    image_dim_variable = data['image_dim']
    image_feature_variable = data['feature']
    max_features = torch.tensor(
        image_feature_variable.shape[1], dtype=torch.int)
    image_label_variable = None
    image_location_variable = data['bbox']
    image_location_variable = image_location_variable[:, :max_features.item(
    ), :4]
    cls_prob = data['cls_prob']

    answers = data['answers_scores']
    is_correct = None

    return {
        'input_ids': bert_input_ids,
        'token_type_ids': bert_input_mask,
        'attention_mask': bert_input_type_ids,
        'masked_lm_labels': masked_lm_labels,
        'image_feature': image_feature_variable,
        'image_location': image_location_variable,
        'masked_image_labels': image_label_variable,
        'obj_labels': cls_prob,
        'matched_label': is_correct,
        'ans': answers,
        'image_dim': image_dim_variable,
        'max_features': max_features,
        # "dataset_name": str(sample_list.dataset_name),
    }

  # def get_optimizer_parameters(self, config):
  #     return get_optimizer_parameters_for_bert(self.model, config)

  # def forward(self, sample_list):
  #     device = registry.get("config").training.device
  #     params = self.get_image_and_text_features(sample_list, device)
  #     if params["visual_feats"] is not None and params["image_dim"] is not None:
  #         device = params["visual_feats"].device
  #         image_mask = (
  #             torch.arange(params["visual_feats"].size(-2))
  #             .expand(*params["visual_feats"].size()[:-1])
  #             .to(device)
  #         )
  #         if len(params["image_dim"].size()) < len(image_mask.size()):
  #             params["image_dim"] = params["image_dim"].unsqueeze(-1)
  #             assert len(params["image_dim"].size()) == len(image_mask.size())
  #         image_mask = image_mask < params["image_dim"]
  #         params["image_attention_mask"] = image_mask.long()
  #     else:
  #         params["image_attention_mask"] = None
  #     if self.config.training_head_type == "pretraining":
  #         output_dict = self.model(
  #             input_ids=params["input_ids"],
  #             token_type_ids=params["token_type_ids"],
  #             attention_mask=params["attention_mask"],
  #             visual_feats=params["visual_feats"],
  #             visual_pos=params["pos"],
  #             visual_attention_mask=params["image_attention_mask"],
  #             masked_lm_labels=params["masked_lm_labels"],
  #             masked_image_labels=params["masked_image_labels"],
  #             obj_labels=params["obj_labels"],
  #             matched_label=params["matched_label"],
  #             ans=params["ans"],
  #             num_features=params["max_features"],
  #             name=params["dataset_name"],
  #         )
  #         loss_key = "{}/{}".format(
  #             sample_list.dataset_name, sample_list.dataset_type
  #         )
  #         output_dict["losses"] = {}
  #         if "masked_lm_loss" in output_dict.keys():
  #             output_dict["losses"][loss_key + "/masked_lm_loss"] = output_dict.pop(
  #                 "masked_lm_loss"
  #             )
  #         if "matched_loss" in output_dict.keys():
  #             output_dict["losses"][loss_key + "/matched_loss"] = output_dict.pop(
  #                 "matched_loss"
  #             )
  #         if "visn_loss" in output_dict.keys():
  #             output_dict["losses"][loss_key + "/visn_loss"] = output_dict.pop(
  #                 "visn_loss"
  #             )
  #         if "answer_loss" in output_dict.keys():
  #             output_dict["losses"][loss_key + "/answer_loss"] = output_dict.pop(
  #                 "answer_loss"
  #             )
  #     else:
  #         output_dict = self.model(
  #             input_ids=params["input_ids"],
  #             token_type_ids=params["token_type_ids"],
  #             attention_mask=params["attention_mask"],
  #             visual_feats=params["visual_feats"],
  #             visual_pos=params["pos"],
  #             visual_attention_mask=params["image_attention_mask"],
  #         )
  #     return output_dict

  def forward_train(self, data):
    params = self.get_image_and_text_features(data)
    params['masked_lm_labels'] = data['input_lm_label_ids']
    # Prepare Mask
    if params['image_feature'] is not None and params['image_dim'] is not None:
      image_mask = (
          torch.arange(params['image_feature'].size(-2)).expand(
              *params['image_feature'].size()[:-1]))
      if len(params['image_dim'].size()) < len(image_mask.size()):
        params['image_dim'] = data['image_dim'].unsqueeze(-1)
        assert len(params['image_dim'].size()) == len(image_mask.size())
      image_mask = image_mask < params['image_dim']
      params['image_attention_mask'] = image_mask.long()
    else:
      params['image_attention_mask'] = None
    output_dict = self.model(
        input_ids=params['input_ids'].cuda(),
        token_type_ids=params['token_type_ids'].cuda(),
        attention_mask=params['attention_mask'].cuda(),
        visual_feats=params['image_feature'].cuda(),
        visual_pos=params['image_location'].cuda(),
        visual_attention_mask=params['image_attention_mask'].cuda(),
        #masked_lm_labels=params["masked_lm_labels"],
        #masked_image_labels=params["masked_image_labels"],
        #obj_labels=params["obj_labels"],
        #matched_label=params["matched_label"],
        #ans=params["ans"],
        #num_features=params["max_features"],
        #name=params["dataset_name"],
    )

    model_output = {
        'scores': output_dict['scores'],
        'target': params['ans'].cuda()
    }
    return model_output
    # losses = F.binary_cross_entropy_with_logits(output_dict['scores'],
    #                                             params['ans'].cuda())
    # return {'losses': losses}
