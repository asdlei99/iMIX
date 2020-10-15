from .base_hook import HookBase
from torch.nn.utils import clip_grad

from .builder import HOOKS


@HOOKS.register_module()
class OptimizerHook(HookBase):

    def __init__(self, grad_clip=None):
        self._grad_clip = grad_clip

    def __clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self._grad_clip)

    def after_iter(self):
        pass
        # self.trainer.optimizer.zero_grad()
        # self.trainer.output['loss'].backward()
        # if self._grad_clip is not None:
        #     grad_norm = self._grad_clip(self.trainer.parameters())
        #     if grad_norm is not None:
        #         self.trainer.log_buffer.push_scalar(
        #             'grad_norm',
        #             float(grad_norm))  #TODO(jinliang) 缺少num_samples
        #
        # self.trainer.optimizer.step()
