import torch

from ..builder import VQA_MODELS, build_backbone, build_combine_layer, build_embedding, build_encoder, build_head
from .base_model import BaseModel


def filter_grads(parameters):
    return [param for param in parameters if param.requires_grad]


@VQA_MODELS.register_module()
class MCAN(BaseModel):

    def __init__(self, embedding, encoder, backbone, combine_model, head):
        super().__init__()
        self.embedding_model = build_embedding(embedding)
        self.encoder_model = build_encoder(encoder)
        self.backbone = build_backbone(backbone)
        self.combine_model = build_combine_layer(combine_model)  # combine text and image
        self.head = build_head(head)  # 包括 classification head， generation head

    def get_optimizer_parameters(self, optimizer_params_lr, training_encoder_lr_multiply):
        combine_layer = self.combine_model
        params = [
            {
                'params': filter_grads(self.embedding_model[0].parameters())
            },
            {
                'params': filter_grads(self.backbone.sga.parameters())
            },
            {
                'params': filter_grads(self.backbone.sga_pool.parameters())
            },
            {
                'params': filter_grads(self.backbone.cbn.parameters()),
                'lr': (optimizer_params_lr * training_encoder_lr_multiply),
            },
            {
                'params': filter_grads(self.embedding_model[-1].parameters())
            },
            {
                'params': filter_grads(combine_layer.parameters())
            },
            {
                'params': filter_grads(self.head.parameters())
            },
            {
                'params': filter_grads(self.encoder_model.parameters())
            },
        ]

        return params

    def process_text_embedding(self, text, text_mask):
        # Get embedding models
        text_embedding_model = self.embedding_model[-1]
        text_embedding_total, text_embedding_vec = text_embedding_model(text, text_mask)
        return text_embedding_total, text_embedding_vec

    def process_feature_embedding(self,
                                  img_feat,
                                  text_embedding_total,
                                  text_embedding_vec,
                                  text_mask,
                                  vextra=None,
                                  batch_size_t=None):
        image_feature_0 = img_feat
        encoded_feature = self.encoder_model(image_feature_0)
        feature_sga, feature_cbn = self.backbone(encoded_feature, text_embedding_total, text_embedding_vec, None,
                                                 text_mask)

        return feature_sga, feature_cbn

    # def forward(self, img_feat, input_ids, text_mask):
    #     ques_feat = self.embedding_model[0](input_ids)
    #     # text_mask = ques_feat.eq(0)
    #     text_embedding_total, text_embedding_vec = self.process_text_embedding(
    #         ques_feat, text_mask)
    #
    #     feature_sga, feature_cbn = self.process_feature_embedding(
    #         img_feat, text_embedding_total, text_embedding_vec[:, 0],
    #         text_mask)
    #
    #     joint_embedding = self.combine_model(feature_sga, feature_cbn,
    #                                          text_embedding_vec[:, 1])
    #
    #     model_output = {"scores": self.head(joint_embedding)}
    #
    #     return model_output
    def __joint_embedding(self, data, **kwargs):
        batch_data = self.preprocess_data(data)

        img_feat = batch_data['feature']
        input_ids = batch_data['input_ids']
        text_mask = batch_data['input_mask']
        # targets = batch_data['answers_scores']

        ques_feat = self.embedding_model[0](input_ids)
        # text_mask = ques_feat.eq(0)
        text_embedding_total, text_embedding_vec = self.process_text_embedding(ques_feat, text_mask)

        feature_sga, feature_cbn = self.process_feature_embedding(img_feat, text_embedding_total,
                                                                  text_embedding_vec[:, 0], text_mask)

        joint_embedding = self.combine_model(feature_sga, feature_cbn, text_embedding_vec[:, 1])
        return joint_embedding

    def forward_train(self, data, **kwargs):

        batch_data = self.preprocess_data(data)
        joint_embedding = self.__joint_embedding(batch_data, **kwargs)

        targets = batch_data['answers_scores']
        model_output = self.head.forward_train(joint_embedding, labels=targets)
        return model_output

    def forward_test(self, data, **kwargs):
        assert not self.training

        batch_data = self.preprocess_data(data)
        joint_embedding = self.__joint_embedding(batch_data, **kwargs)
        model_output = {'scores': self.head.forward_test(joint_embedding)}
        return model_output

    def preprocess_data(self, batched_inputs):
        from imix.engine.organizer import is_by_iter
        if is_by_iter():
            batched_inputs = list2dict(batched_inputs)

        img_feat = batched_inputs['feature']
        input_ids = batched_inputs['input_ids']
        input_mask = batched_inputs['input_mask']

        b, c, h, w = img_feat.shape
        feat = img_feat.view(b, c, -1)
        padded_feat = torch.zeros((b, c, 1024), dtype=torch.float)
        padded_feat[:, :, :h * w] = feat
        feat = padded_feat.unsqueeze(-1)
        # feat = feat.squeeze(0)
        feat = feat.cuda()

        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

        batched_inputs['feature'] = feat
        batched_inputs['input_ids'] = input_ids
        batched_inputs['input_mask'] = ~input_mask  # TODO(jinliang):lixiaochuan

        if self.training:
            answers_scores = batched_inputs['answers_scores']
            answers_scores = answers_scores.cuda()
            batched_inputs['answers_scores'] = answers_scores

        return batched_inputs


def list2dict(batched_inputs):  # TODO(jinliang):
    batch_size = len(batched_inputs)
    img_feats = torch.zeros((batch_size, *batched_inputs[0]['feature'].shape), dtype=batched_inputs[0]['feature'].dtype)
    input_ids = torch.zeros((batch_size, *batched_inputs[0]['input_ids'].shape),
                            dtype=batched_inputs[0]['input_ids'].dtype)
    answers_scores = torch.zeros((batch_size, *batched_inputs[0]['answers_scores'].shape),
                                 dtype=batched_inputs[0]['answers_scores'].dtype)
    input_mask = torch.zeros((batch_size, *batched_inputs[0]['input_mask'].shape),
                             dtype=batched_inputs[0]['input_mask'].dtype)
    question_id = torch.zeros([batch_size], dtype=torch.int32)
    for idx in range(batch_size):
        img_feats[idx] = batched_inputs[idx]['feature']
        input_ids[idx] = batched_inputs[idx]['input_ids']
        answers_scores[idx] = batched_inputs[idx]['answers_scores']
        input_mask[idx] = batched_inputs[idx]['input_mask']
        question_id[idx] = batched_inputs[idx]['question_id']

    batch_data = dict()

    batch_data['feature'] = img_feats
    batch_data['input_ids'] = input_ids
    batch_data['answers_scores'] = answers_scores
    batch_data['input_mask'] = input_mask
    batch_data['question_id'] = question_id

    return batch_data
