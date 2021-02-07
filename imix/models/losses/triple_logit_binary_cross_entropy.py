import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
import torch
from .base_loss import BaseLoss


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

  def forward(self, predict_scores, target):
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

  def forward(self, predict_scores, target):
    return self.loss_fn(predict_scores, target)

  def __str__(self):
    return 'cross_entropy_loss'


@LOSSES.register_module()
class LogitBinaryCrossEntropy(BaseLoss):
  """Returns Binary Cross Entropy for logits.

  Attention:
      `Key`: logit_bce
  """

  def __init__(self):
    super().__init__(loss_name=str(self))

  def forward(self, scores, targets):
    """Calculates and returns the binary cross entropy for logits.

    Args:
        sample_list (SampleList): SampleList containing `targets` attribute.
        model_output (Dict): Model output containing `scores` attribute.

    Returns:
        torch.FloatTensor: Float value for loss.
    """
    # scores = model_output["scores"]
    # targets = sample_list["targets"]
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
    super().__init__()
    self.one = torch.Tensor([1.0])

  def forward(self, sample_list, model_output):
    scores = model_output['scores']
    targets = sample_list['targets']
    loss_mask = sample_list['train_loss_mask']
    assert scores.dim() == 3 and loss_mask.dim() == 2

    losses = F.binary_cross_entropy_with_logits(
        scores, targets, reduction='none')
    losses *= loss_mask.unsqueeze(-1)

    count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
    loss = torch.sum(losses) / count
    return loss
