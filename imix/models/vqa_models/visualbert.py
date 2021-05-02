import torch

from ..builder import VQA_MODELS
from ..encoder import VisualBERTForClassification, VisualBERTForPretraining
from .base_model import BaseModel


def transform_to_batch_sequence(tensor):
    if tensor is not None:
        if len(tensor.size()) == 2:
            return tensor
        else:
            assert len(tensor.size()) == 3
            return tensor.contiguous().view(-1, tensor.size(-1))
    else:
        return None


def transform_to_batch_sequence_dim(tensor):
    if tensor is not None:
        if len(tensor.size()) == 3:
            return tensor
        else:
            assert len(tensor.size()) == 4
            return tensor.contiguous().view(-1, tensor.size(-2), tensor.size(-1))
    else:
        return None


@VQA_MODELS.register_module()
class VisualBERT(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()

        params = kwargs['params']
        self.special_visual_initialize = params['special_visual_initialize']
        freeze_base = params['freeze_base']
        self.training_head_type = params['training_head_type']
        if self.training_head_type == 'pretraining':
            self.model = VisualBERTForPretraining(**params)
        else:
            self.model = VisualBERTForClassification(**params)

        if self.special_visual_initialize:
            self.model.bert.embeddings.initialize_visual_from_pretrained()

        if freeze_base:
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def flatten(self, data, to_be_flattened=None, to_be_flattened_dim=None):
        if to_be_flattened is None:
            to_be_flattened = {}
        if to_be_flattened_dim is None:
            to_be_flattened_dim = {}
        for key in to_be_flattened:
            # Make sure these keys are present or otherwise set these keys to None
            # sample_list[key] = getattr(sample_list, key, None)
            data[key] = transform_to_batch_sequence(data[key])
        for key in to_be_flattened_dim:
            # sample_list[key] = getattr(sample_list, key, None)
            data[key] = transform_to_batch_sequence_dim(data[key])

        if data['visual_embeddings_type'] is None:
            if data['image_mask'] is not None:
                data['visual_embeddings_type'] = torch.zeros_like(data['image_mask'], dtype=torch.long)

        if data['image_mask'] is not None:
            attention_mask = torch.cat((data['input_mask'], data['image_mask']), dim=-1)
            if data['input_lm_label_ids'] is not None:
                assert data['input_lm_label_ids'].size(-1) == data['input_mask'].size(-1)
                new_lm_labels = torch.ones_like(attention_mask) * -1
                size_masked_lm_labels = data['input_lm_label_ids'].size()
                assert len(size_masked_lm_labels) == 2
                new_lm_labels[:size_masked_lm_labels[0], :size_masked_lm_labels[1]] = data['input_lm_label_ids']
                data['input_lm_label_ids'] = new_lm_labels
        else:
            attention_mask = data['input_mask']

        data['attention_mask'] = attention_mask

        return data

    # def get_optimizer_parameters(self, config):
    #     return get_optimizer_parameters_for_bert(self.model, config)

    def flatten_for_bert(self, sample_list, **kwargs):
        to_be_flattened = [
            'input_ids',
            'input_segment',
            'input_mask',
            'image_mask',
            'input_lm_label_ids',
            'position_embeddings_visual',
            'visual_embeddings_type',
        ]
        to_be_flattened_dim = ['image_text_alignment', 'features']

        # We want to convert everything into: batch x sequence_length x (dim).
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    # def update_sample_list_based_on_head(self, data):
    #     bert_input_ids = data['input_ids']
    #     bert_input_mask = data['input_mask']
    #     bert_input_type_ids = data['input_segment']

    # if self.config.training_head_type == 'nlvr2':
    #     bert_input_ids = torch.cat([bert_input_ids, bert_input_ids])
    #     bert_input_mask = torch.cat([bert_input_mask, bert_input_mask])
    #     bert_input_type_ids = torch.cat(
    #         [bert_input_type_ids, bert_input_type_ids])
    #
    #     # image input
    #     img0 = getattr(sample_list, 'img0', {})
    #     image_info = getattr(img0, 'image_info_0', {})
    #     image_dim_variable_0 = getattr(image_info, 'max_features', None)
    #     image_feat_variable_0 = getattr(img0, 'image_feature_0', None)
    #
    #     img1 = getattr(sample_list, 'img1', {})
    #     image_info = getattr(img1, 'image_info_0', {})
    #     image_dim_variable_1 = getattr(image_info, 'max_features', None)
    #     image_feat_variable_1 = getattr(img1, 'image_feature_0', None)
    #
    #     image_feat_variable = torch.cat(
    #         [image_feat_variable_0, image_feat_variable_1])
    #     image_dim_variable = torch.cat(
    #         [image_dim_variable_0, image_dim_variable_1])
    # else:
    # image_info = getattr(data, 'image_info_0', {})
    # image_dim_variable = getattr(image_info, 'max_features', None)
    # image_feat_variable = getattr(sample_list, 'image_feature_0', None)
    # image_feat_variable = data['features']
    #
    # sample_list.visual_embeddings = image_feat_variable
    # sample_list.image_dim = image_dim_variable
    # sample_list.input_ids = bert_input_ids
    # sample_list.input_mask = bert_input_mask
    # sample_list.token_type_ids = bert_input_type_ids
    # return sample_list

    def add_custom_params(self, data):
        visual_embeddings = data['features']
        image_dim = data['image_dim']
        # image_feat_variable = batch x ( num_choice x ) image_feature_length x dim
        # Prepare Mask
        if visual_embeddings is not None:
            image_mask = (torch.arange(visual_embeddings.size(-2)).expand(*visual_embeddings.size()[:-1]))
            if len(image_dim.size()) < len(image_mask.size()):
                image_dim = image_dim.unsqueeze(-1)
            assert len(image_dim.size()) == len(image_mask.size())
            image_mask = image_mask.to(device=image_dim.device)
            image_mask = image_mask < image_dim
            data['image_mask'] = image_mask.long()
        else:
            data['image_mask'] = None

        data['position_embeddings_visual'] = None
        data['image_text_alignment'] = None
        data['visual_embeddings_type'] = None
        return data

    # Backward compatibility for code from original VisualBERT
    @classmethod
    def format_state_key(cls, key):
        return (key.replace('bert.bert', 'model.bert').replace('bert.cls',
                                                               'model.cls').replace('bert.classifier',
                                                                                    'model.classifier'))

    def forward_train(self, data, *args, **kwargs):
        # data = self.update_sample_list_based_on_head(data)
        data = self.add_custom_params(data)
        data = self.flatten_for_bert(data)

        output_dict = self.model(
            data['input_ids'].cuda(),
            data['input_mask'].cuda(),
            data['attention_mask'].cuda(),
            data['input_segment'].cuda(),
            data['features'].cuda(),
            None,  # data['position_embeddings_visual'],
            data['visual_embeddings_type'].cuda(),
            None,  # data['image_text_alignment'],
            data['input_lm_label_ids'].cuda())
        # losses = F.binary_cross_entropy_with_logits(output_dict['scores'],
        #                                             data['answers_scores'].cuda())
        # return {'losses': losses}

        model_output = {'scores': output_dict['scores'], 'target': data['answers_scores'].cuda()}
        return model_output
