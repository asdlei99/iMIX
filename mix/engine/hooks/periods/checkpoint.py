from mix.utils.checkpoint import PeriodicCheckpointer
from ..base_hook import HookBase
from ..builder import HOOKS


@HOOKS.register_module()
class CheckPointHook(PeriodicCheckpointer, HookBase):

    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_iter(self):
        self.step(self.trainer.iter)
