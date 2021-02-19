from typing import Dict, Tuple
from collections import OrderedDict
import torch
import torch.distributed as dist
from abc import abstractmethod, ABCMeta
from ..builder import build_loss


class BaseLoss(torch.nn.Module, metaclass=ABCMeta):
    loss_name = 'base_loss'

    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name

    @abstractmethod
    def forward(self, *args, **kwargs):
        # return NotImplementedError
        pass

    # def loss(self, *args, **kwargs):
    #     pass

    # def loss(self, scores, targets):
    #     losses = {str(self): self.forward(scores, targets)}
    #     loss, losses_log = self.parse_losses(losses=losses)
    #     output = losses_log
    #     output['loss'] = loss
    #     return output

    # def __call__(self, *args, **kwargs):
    #     pass


class Losser:
    def __init__(self, losses_cfg):
        self._loss_list: list = self.build_losses(losses_cfg)

    @classmethod
    def build_losses(cls, losses_cfg):
        loss_objs = build_loss(losses_cfg)
        return loss_objs if isinstance(loss_objs, list) else [loss_objs]

    def __call__(self, model_output: Dict):
        losses_dict = {}
        for loss_obj in self._loss_list:
            # losses = {str(loss_obj): loss_obj.forward(*args, **kwargs)}
            losses = {str(loss_obj): loss_obj.forward(model_output)}
            losses_dict.update(losses)

        losses, losses_log = self.parse_losses(losses=losses_dict)
        output = losses_log
        output['loss'] = losses
        return output

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
