from abc import ABCMeta

import torch.nn as nn
from apex.normalization.fused_layer_norm import FusedLayerNorm
import torch
# class BaseModel(nn.Module):
#
#   def __init__(self):
#     super().__init__()
#
#   def forward(self, data, **kwargs):
#     if self.training:
#       losses = self.forward_train(data, **kwargs)
#       loss, losses_log = self._parse_losses(losses=losses)
#       model_outputs = dict()
#       model_outputs.update(losses_log)
#       model_outputs['loss'] = loss
#       return model_outputs
#     else:
#       return self.forward_test(data, **kwargs)
#
#   def forward_train(self, data, **kwargs):
#     pass
#
#   def forward_test(self, data, **kwargs):
#     pass
#
#   @staticmethod
#   def _parse_losses(losses: Dict) -> Tuple[torch.Tensor, Dict]:
#
#     losses_log = OrderedDict()
#     for name, value in losses.items():
#       losses_log[name] = value.mean()
#
#     loss = sum(v for k, v in losses_log.items() if 'loss' in k)
#     losses_log['loss'] = loss
#     for name, value in losses_log.items():
#       if dist.is_available() and dist.is_initialized():
#         loss_value = value.data.clone()
#         value = loss_value.div_(dist.get_world_size())
#         dist.all_reduce(value)
#
#       losses_log[name] = value.item()
#     return loss, losses_log


class BaseModel(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    # def forward(self, data, **kwargs):
    #     if self.training:
    #         losses = self.forward_train(data, **kwargs)
    #         loss, losses_log = self._parse_losses(losses=losses)
    #         model_outputs = dict()
    #         model_outputs.update(losses_log)
    #         model_outputs['loss'] = loss
    #         return model_outputs
    #     else:
    #         return self.forward_test(data, **kwargs)

    def forward(self, data, **kwargs):
        if self.training:
            return self.forward_train(data, **kwargs)
        else:
            return self.forward_test(data, **kwargs)

    def forward_train(self, data, **kwargs):
        pass

    def forward_test(self, data, **kwargs):
        pass


class UniterBaseModel(BaseModel):
    """An abstract class to handle weights initialization and a simple
    interface for dowloading and loading pretrained models."""

    def __init__(self, *inputs, **kwargs):
        super().__init__()

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def set_dropout(self, drop_p):
        for name, module in self.named_modules():
            # we might want to tune dropout for smaller dataset
            if isinstance(module, torch.nn.Dropout):
                if module.p != drop_p:
                    module.p = drop_p
