from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast  # TODO(jinliang)

from ..builder import VQA_MODELS, build_backbone, build_combine_layer, build_embedding, build_encoder, build_head


def filter_grads(parameters):
    return [param for param in parameters if param.requires_grad]


@VQA_MODELS.register_module()
class MCAN(nn.Module):

    def __init__(self, embedding, encoder, backbone, combine_model, head):
        # super().__init__()
        super(MCAN, self).__init__()
        self.embedding_model = build_embedding(embedding)
        self.encoder_model = build_encoder(encoder)
        self.backbone = build_backbone(backbone)
        self.combine_model = build_combine_layer(combine_model)  ###combine text and image
        self.head = build_head(head)  ###包括 classification head， generation head

        # self.init_weights()
        self.trip_loss = TripleLogitBinaryCrossEntropy()

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

    def forward(self, batch_data):  # TODO(jinliang): imitate Det2
        from imix.engine.organizer import is_multi_gpus_mixed_precision
        with autocast(enabled=is_multi_gpus_mixed_precision()):
            if not self.training:
                return self.inference(batch_data)
            batch_data = self.preprocess_data(batch_data)

            img_feat = batch_data['feature']
            input_ids = batch_data['input_ids']
            text_mask = batch_data['input_mask']
            targets = batch_data['answers_scores']

            ques_feat = self.embedding_model[0](input_ids)
            # text_mask = ques_feat.eq(0)
            text_embedding_total, text_embedding_vec = self.process_text_embedding(ques_feat, text_mask)

            feature_sga, feature_cbn = self.process_feature_embedding(img_feat, text_embedding_total,
                                                                      text_embedding_vec[:, 0], text_mask)

            joint_embedding = self.combine_model(feature_sga, feature_cbn, text_embedding_vec[:, 1])

            model_output = {'scores': self.head(joint_embedding), 'target': targets}

            return model_output

        # loss = self.trip_loss(targets, model_output)
        # losses = dict(bce_loss=loss)
        #
        # # loss, log_vars = self._parse_losses(losses)
        # #
        # # outputs = dict(
        # #     loss=loss,
        # #     log_vars=log_vars,
        # #     num_samples=len(batch_data['feature']))
        #
        # outputs = dict(
        #     loss=loss,
        #     # log_vars=losses,
        #     num_samples=len(batch_data['feature']))
        # outputs.update(losses)
        # return outputs

    def losses(self, output, target):
        loss = self.trip_loss(output, target)
        losses = dict(bce_loss=loss)

        # loss, log_vars = self._parse_losses(losses)
        #
        # outputs = dict(
        #     loss=loss,
        #     log_vars=log_vars,
        #     num_samples=len(batch_data['feature']))

        outputs = dict(
            loss=loss,
            # log_vars=losses,
            num_samples=len(target))
        outputs.update(losses)
        return outputs

    def inference(self, batch_data):
        assert not self.training

        batch_data = self.preprocess_data(batch_data)

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

        model_output = {'scores': self.head(joint_embedding)}
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

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


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

    # def train_step(self, data, optimizer):
    #     """The iteration step during training.
    #
    #     This method defines an iteration step during training, except for the
    #     back propagation and optimizer updating, which are done in an optimizer
    #     hook. Note that in some complicated cases or models, the whole process
    #     including back propagation and optimizer updating is also defined in
    #     this method, such as GAN.
    #
    #     Args:
    #         data (dict): The output of dataloader.
    #         optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
    #             runner is passed to ``train_step()``. This argument is unused
    #             and reserved.
    #
    #     Returns:
    #         dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
    #             ``num_samples``.
    #
    #             - ``loss`` is a tensor for back propagation, which can be a \
    #             weighted sum of multiple losses.
    #             - ``log_vars`` contains all the variables to be sent to the
    #             logger.
    #             - ``num_samples`` indicates the batch size (when the model is \
    #             DDP, it means the batch size on each GPU), which is used for \
    #             averaging the logs.
    #     """
    #     # losses = self(**data)
    #     model_output = self(data)
    #     targets = data['answers_scores']
    #     trip_loss = TripleLogitBinaryCrossEntropy().cuda()
    #     loss = trip_loss(targets, model_output)
    #     losses = dict(bce_loss=loss)
    #     loss, log_vars = self._parse_losses(losses)
    #
    #     outputs = dict(
    #         loss=loss, log_vars=log_vars, num_samples=len(data['feature']))
    #
    #     return outputs


# class TripleLogitBinaryCrossEntropy(nn.Module):
#     """This is used for Three-branch fusion only.
#
#     We predict scores and compute cross entropy loss for each of branches.
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, model_output, targets):
#         """Calculates and returns the binary cross entropy for logits
#         Args:
#             sample_list (SampleList): SampleList containing `targets` attribute.
#             model_output (Dict): Model output containing `scores` attribute.
#         Returns:
#             torch.FloatTensor: Float value for loss.
#         """
#         scores = model_output['scores']
#
#         if scores.dim() == 3:
#             loss = (
#                 func.binary_cross_entropy_with_logits(
#                     scores[:, 0], targets, reduction='mean') +
#                 func.binary_cross_entropy_with_logits(
#                     scores[:, 1], targets, reduction='mean') +
#                 func.binary_cross_entropy_with_logits(
#                     scores[:, 2], targets, reduction='mean'))
#         else:
#             loss = func.binary_cross_entropy_with_logits(
#                 scores, targets, reduction='mean')
#
#         return loss * targets.size(-1)
