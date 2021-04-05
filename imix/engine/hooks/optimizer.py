from .base_hook import HookBase, PriorityStatus
from torch.nn.utils import clip_grad
from .builder import HOOKS
from torch.cuda.amp.grad_scaler import GradScaler

# @HOOKS.register_module()
# class OptimizerHook(HookBase):
#
#     def __init__(self, grad_clip=None):
#         self._grad_clip = grad_clip
#
#     # def _clip_grads(self, params):  # TODO(jinliang):jinliang_copy
#     #     params = list(
#     #         filter(lambda p: p.requires_grad and p.grad is not None, params))
#     #     if len(params) > 0:
#     #         return clip_grad.clip_grad_norm_(params, **self._grad_clip)
#
#     def _clip_grad_norm(self):
#         params = list(
#             filter(lambda p: p.requires_grad and p.grad is not None,
#                    self.trainer.parameters()))
#         clip_norm_params = list(
#             filter(lambda parm: parm.requires_grad and parm.grad is not None,self.trainer.parameters())
#         )
#
#         if len(params) == 0:
#             return
#         else:
#             grad_norm = clip_grad.clip_grad_norm_(params, **self._grad_clip)
#
#         if grad_norm is not None:
#             self.trainer.log_buffer.push_scalar('grad_norm', float(grad_norm))
#
#     def after_train_iter(self):  # TODO(jinliang):jinliang_imitate
#         self.trainer.output['loss'].backward()
#         if self._grad_clip is not None:
#             self._clip_grad_norm()
#         self.trainer.optimizer.step()
#
#     def before_train_iter(self):
#         self.trainer.optimizer.zero_grad()


@HOOKS.register_module()
class OptimizerHook(HookBase):

    def __init__(self, grad_clip=None):
        self._grad_clip = grad_clip
        self._level = PriorityStatus.HIGH

    def _clip_grad_norm(self) -> None:
        clip_norm_params = list(
            filter(lambda parm: parm.requires_grad and parm.grad is not None, self.trainer.parameters()))
        if len(clip_norm_params) == 0:
            return
        else:
            grad_norm = clip_grad.clip_grad_norm_(clip_norm_params, **self._grad_clip)
            self.trainer.log_buffer.push_scalar('grad_norm', float(grad_norm))  # TODO(jinliang) -> test

    def after_train_iter(self):  # TODO(jinliang):jinliang_imitate
        self.trainer.output['loss'].backward()
        if self._grad_clip is not None:
            self._clip_grad_norm()
        self.trainer.optimizer.step()

    def before_train_iter(self):
        self.trainer.optimizer.zero_grad()

    @property
    def level(self):
        return self._level


@HOOKS.register_module()
class Fp16OptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None, grad_scaler_config=None):
        super().__init__(grad_clip)
        self._grad_scaler_config = grad_scaler_config
        self._scaler = None
        self.__level = PriorityStatus.HIGH

    def before_train(self):
        if self._grad_scaler_config is None:
            self._scaler = GradScaler()
        else:
            self._scaler = GradScaler(**self._grad_scaler_config)

    def after_train_iter(self):
        loss = self.trainer.output['loss']
        self._scaler.scale(loss).backward()
        if self._grad_clip is not None:
            self._scaler.unscale_(self.trainer.optimizer)
            self._clip_grad_norm()
        self._scaler.step(self.trainer.optimizer)
        self._scaler.update()
