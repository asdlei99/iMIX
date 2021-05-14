import torch.nn.functional as F

from ..builder import LOSSES
from .base_loss import BaseLoss


@LOSSES.register_module()
class UNITERCrossEntropyLoss(BaseLoss):
    loss_name = 'UNITER_cross_entropy_loss'

    def __init__(self, **kwargs):
        super().__init__(loss_name=str(self))
        if kwargs['reduction'] is not None:
            self.reduction = kwargs['reduction']
        else:
            assert False

    def forward(self, model_output):
        rank_scores, targets = model_output['scores'], model_output['targets']
        # TODO: confirm the targets dim in dataloader
        return F.cross_entropy(rank_scores, targets, reduction=self.reduction)

    def __str__(self):
        return self.loss_name
