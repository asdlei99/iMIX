from typing import Dict, Tuple
from collections import OrderedDict
import torch
import torch.distributed as dist
from abc import abstractmethod, ABCMeta


class BaseLoss(torch.nn.Module, metaclass=ABCMeta):
  loss_name = 'base_loss'

  def __init__(self, loss_name, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.loss_name = loss_name

  @classmethod
  def parse_losses(cls, losses: Dict) -> Tuple[torch.Tensor, Dict]:

    losses_log = OrderedDict()
    for name, value in losses.items():
      losses_log[name] = value.mean()

    loss = sum(v for k, v in losses_log.items() if 'loss' in k)
    losses_log['total_loss'] = loss
    for name, value in losses_log.items():
      if dist.is_available() and dist.is_initialized():
        loss_value = value.data.clone()
        value = loss_value.div_(dist.get_world_size())
        dist.all_reduce(value)

      losses_log[name] = value.item()
    return loss, losses_log

  @abstractmethod
  def forward(self, *args, **kwargs):
    # return NotImplementedError
    pass

  def loss(self, scores, targets):
    losses = {str(self): self.forward(scores, targets)}
    loss, losses_log = self.parse_losses(losses=losses)
    output = losses_log
    output['loss'] = loss
    return output
