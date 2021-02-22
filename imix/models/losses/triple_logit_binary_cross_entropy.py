import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
import torch
from .base_loss import BaseLoss
from torch.nn import CrossEntropyLoss as TorchCrossEntropyLoss
from torch.nn import SmoothL1Loss as TorchSmoothL1Loss


@LOSSES.register_module()
class TripleLogitBinaryCrossEntropy(BaseLoss):
    """This is used for Three-branch fusion only.

    We predict scores and compute cross entropy loss for each of branches.
    """

    def __init__(self):
        super().__init__(loss_name=str(self))

        # def forward(self, model_output, targets):
        #     """Calculates and returns the binary cross entropy for logits
        #     Args:
        #         sample_list (SampleList): SampleList containing `targets` attribute.
        #         model_output (Dict): Model output containing `scores` attribute.
        #     Returns:
        #         torch.FloatTensor: Float value for loss.
        #     """
        #     scores = model_output['scores']
        #
        #     if scores.dim() == 3:
        #         loss = (
        #                 F.binary_cross_entropy_with_logits(
        #                     scores[:, 0], targets, reduction='mean') +
        #                 F.binary_cross_entropy_with_logits(
        #                     scores[:, 1], targets, reduction='mean') +
        #                 F.binary_cross_entropy_with_logits(
        #                     scores[:, 2], targets, reduction='mean'))
        #     else:
        #         loss = F.binary_cross_entropy_with_logits(
        #             scores, targets, reduction='mean')
        #
        #     return loss * targets.size(-1)

    def forward(self, predict_scores, target):
        """Calculates and returns the binary cross entropy for logits
            Args:
                sample_list (SampleList): SampleList containing `targets` attribute.
                model_output (Dict): Model output containing `scores` attribute.
            Returns:
                torch.FloatTensor: Float value for loss.
            """
        scores = predict_scores

        if scores.dim() == 3:
            loss = (
                    F.binary_cross_entropy_with_logits(
                        scores[:, 0], target, reduction='mean') +
                    F.binary_cross_entropy_with_logits(
                        scores[:, 1], target, reduction='mean') +
                    F.binary_cross_entropy_with_logits(
                        scores[:, 2], target, reduction='mean'))
        else:
            loss = F.binary_cross_entropy_with_logits(
                scores, target, reduction='mean')

        return loss * target.size(-1)

    def __str__(self):
        return 'triple_logit_binary_cross_entropy_loss'


@LOSSES.register_module()
class BinaryCrossEntropyWithLogits(BaseLoss):

    def __init__(self):
        super().__init__(loss_name=str(self))

    def forward(self, model_output):
        predict_scores, target = model_output['scores'], model_output['target']
        return F.binary_cross_entropy_with_logits(predict_scores, target)

    def __str__(self):
        return 'binary_cross_entropy_with_logits_loss'


@LOSSES.register_module()
class CrossEntropyLoss(BaseLoss):

    def __init__(self, params=None):
        super().__init__(loss_name=str(self))
        if params is None:
            params = {}
        self.loss_fn = nn.CrossEntropyLoss(**params)

    # def forward(self, sample_list, model_output):
    #     return self.loss_fn(model_output['scores'], sample_list.targets)

    def forward(self, model_output):
        predict_scores, target = model_output['scores'], model_output['target']
        return self.loss_fn(predict_scores, target)

    def __str__(self):
        return 'cross_entropy_loss'

@LOSSES.register_module()
class OBJCrossEntropyLoss(BaseLoss):

    def __init__(self, params=None):
        super().__init__(loss_name=str(self))
        if params is None:
            params = {}
        self.loss_fn = nn.CrossEntropyLoss(**params)

    # def forward(self, sample_list, model_output):
    #     return self.loss_fn(model_output['scores'], sample_list.targets)

    def forward(self, model_output):
        predict_scores, target = model_output['obj_scores'], model_output['obj_target']
        return self.loss_fn(predict_scores, target)

    def __str__(self):
        return 'obj_cross_entropy_loss'


@LOSSES.register_module()
class LogitBinaryCrossEntropy(BaseLoss):
    """Returns Binary Cross Entropy for logits.

    Attention:
        `Key`: logit_bce
    """

    def __init__(self):
        super().__init__(loss_name=str(self))

    def forward(self, model_output):
        """Calculates and returns the binary cross entropy for logits.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.
        """
        # scores = model_output["scores"]
        # targets = sample_list["targets"]
        scores, targets = model_output['scores'], model_output['target']
        loss = F.binary_cross_entropy_with_logits(scores, targets, reduction='mean')

        return loss * targets.size(1)

    def __str__(self):
        return 'logit_binary_cross_entropy_loss'


@LOSSES.register_module()
class CaptionCrossEntropyLoss(BaseLoss):

    def __init__(self):
        super().__init__(loss_name=str(self))

    def forward(self, sample_list, model_output):
        """Calculates and returns the cross entropy loss for captions.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.
        """
        scores = model_output['scores']
        targets = sample_list['targets']

        # If no captions(test dataset) then assume decode length to be uniform
        if hasattr(sample_list, 'caption_len'):
            caption_lengths, _ = sample_list.caption_len.sort(dim=0, descending=True)
            decode_lengths = (caption_lengths - 1).tolist()
        else:
            decode_lengths = [targets.size(1)] * targets.size(0)
        if torch.__version__ >= '1.1':
            scores = pack_padded_sequence(
                scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True).data
        else:
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(
                targets, decode_lengths, batch_first=True)

        loss = F.cross_entropy(scores, targets)

        return loss


@LOSSES.register_module()
class M4CDecodingBCEWithMaskLoss(BaseLoss):

    def __init__(self):
        super().__init__(loss_name=str(self))
        self.one = torch.Tensor([1.0])

    def __str__(self):
        return 'M4CDecodingBCEWithMaskLoss'

    def forward(self, model_output):

        scores = model_output['scores']
        targets = model_output['target']
        loss_mask = model_output['train_loss_mask']
        assert scores.dim() == 3 and loss_mask.dim() == 2

        losses = F.binary_cross_entropy_with_logits(
            scores, targets, reduction='none')
        losses *= loss_mask.unsqueeze(-1)

        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        loss = torch.sum(losses) / count
        return loss

@LOSSES.register_module()
class LXMERTPreTrainLossV0(BaseLoss):

    def __init__(self, visual_losses, visual_loss_config, vocab_size, num_answers):
        super().__init__(loss_name=str(self))
        self.loss_fct_cls = TorchCrossEntropyLoss(ignore_index=-1)
        self.loss_fcts_feat = {
            'l2': TorchSmoothL1Loss(reduction='none'),
            'ce': TorchCrossEntropyLoss(ignore_index=-1, reduction='none')
        }
        self.visual_losses = visual_losses.split(",")
        self.visual_loss_config = visual_loss_config
        self.vocab_size = vocab_size
        self.num_answers = num_answers

    def forward(self, model_output):
        scores = model_output['scores']
        target = model_output['target']
        lang_prediction_scores = scores["lang_prediction_scores"]
        cross_relationship_score = scores["cross_relationship_score"]
        visn_prediction_scores_dict = scores["visn_prediction_scores_dict"]
        answer_score = scores["answer_score"]
        masked_lm_labels = target["masked_lm_labels"]
        matched_label = target["matched_label"]
        obj_labels = target["obj_labels"]
        ans = target["ans"]

        total_loss = 0.
        losses = ()

        masked_lm_loss = self.loss_fct_cls(lang_prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        total_loss += masked_lm_loss
        losses += (masked_lm_loss.detach(),)

        matched_loss = self.loss_fct_cls(cross_relationship_score.view(-1, 2), matched_label.view(-1))
        total_loss += matched_loss
        losses += (matched_loss.detach(),)

        total_visn_loss = 0.
        for key in self.visual_losses:
            label, mask_conf = obj_labels[key]
            output_dim, loss_fct_name, label_shape, weight = self.visual_loss_config[key]
            visn_loss_fct = self.loss_fcts_feat[loss_fct_name]
            visn_prediction_scores = visn_prediction_scores_dict[key]
            visn_loss = visn_loss_fct(
                visn_prediction_scores.view(-1, output_dim),
                label.view(*label_shape),
            )
            if visn_loss.dim() > 1:  # Regression Losses
                visn_loss = visn_loss.mean(1)
            visn_loss = (visn_loss * mask_conf.view(-1)).mean() * weight
            total_visn_loss += visn_loss
            losses += (visn_loss.detach(),)
        total_loss += total_visn_loss

        answer_loss = self.loss_fct_cls(answer_score.view(-1, self.num_answers), ans.view(-1))

        total_loss += answer_loss
        losses += (answer_loss.detach(),)

        return total_loss #, torch.stack(losses).unsqueeze(0), answer_score.detach()

    def __str__(self):
        return 'lxmert_pretrain_loss_v0'

