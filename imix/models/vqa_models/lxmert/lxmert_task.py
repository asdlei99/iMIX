from imix.models.builder import VQA_MODELS
import torch
import copy
'''
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    # BertLayerNorm,
    BertPreTrainedModel,
)
'''
from .lxmert import LXMERTForPretraining
from imix.models.vqa_models.base_model import BaseModel
from .lxmert_qa_answer_table import load_lxmert_qa
import json
from .lxmert import ClassificationModel


@VQA_MODELS.register_module()
class LXMERT(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()

        args = kwargs['params']
        freeze_base = args['freeze_base']
        training_head_type = args['training_head_type']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if training_head_type == 'pretraining':
            self.model = LXMERTForPretraining(**args)
            self.forward_train = self.forward_train_pretrain
        else:
            self.model = ClassificationModel(**args)
            pretrained_path = args['pretrained_path']
            if pretrained_path is not None:
                if training_head_type in ['vqa2', 'gqa']:
                    self.label2ans = json.load(open(args.label2ans_path))
                    load_lxmert_qa(pretrained_path, self.model, label2ans=self.label2ans)
                elif training_head_type == 'nlvr2':
                    self.model.lxrt_encoder.load(pretrained_path)

        if freeze_base:
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def forward_train(self, data, **kwargs):
        # ques_id = data['ques_id'].to(self.device)
        feats = data['feats'].to(self.device)
        boxes = data['boxes'].to(self.device)
        sent = data['ques']
        target = data['target'].to(self.device)

        output_dict = self.model(feats, boxes, sent)

        model_output = {
            'scores': output_dict['scores'],
            'target': target,
        }
        return model_output

    def forward_test(self, data):
        model_output = self.forward_train(data)
        return model_output

    def forward_train_pretrain(self, data):
        params = copy.deepcopy(data)
        if params.get('feats') is not None and params.get('image_dim') is not None:
            image_mask = (torch.arange(params['feats'].size(-2)).expand(*params['feats'].size()[:-1]))
            if len(params['image_dim'].size()) < len(image_mask.size()):
                params['image_dim'] = data['image_dim'].unsqueeze(-1)
                assert len(params['image_dim'].size()) == len(image_mask.size())
            image_mask = image_mask < params['image_dim']
            params['visual_attention_mask'] = image_mask.long()
        else:
            params['visual_attention_mask'] = None
        output_dict = self.model(
            input_ids=params['input_ids'].cuda(),
            token_type_ids=params['segment_ids'].cuda(),
            attention_mask=params['input_mask'].cuda(),
            visual_feats=params['feats'].cuda(),
            visual_pos=params['pos'].cuda(),
            visual_attention_mask=params['visual_attention_mask'].cuda()
            if params['visual_attention_mask'] is not None else params['visual_attention_mask'],
        )
        target_dict = {
            'masked_lm_labels': params['lm_label_ids'].cuda(),
            'matched_label': params['is_matched'].cuda(),
            'ans': params['ans'].cuda(),
            'obj_labels': {
                'obj': (params['det_obj_labels'].cuda(), params['det_obj_confs'].cuda()),
                'attr': (params['det_attr_labels'].cuda(), params['det_attr_confs'].cuda()),
                'feat': (params['det_feat'].cuda(), params['det_feat_mask'].cuda()),
            }
        }
        model_output = {'scores': output_dict, 'target': target_dict}
        return model_output
