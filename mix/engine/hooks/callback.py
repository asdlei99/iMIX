from .base_hook import HookBase

from .builder import HOOKS


@HOOKS.register_module()
class CallBackHook(HookBase):

    def __init__(self,
                 *,
                 before_train=None,
                 after_train=None,
                 before_iter=None,
                 after_iter=None):
        self._before_iter = before_iter
        self._before_train = before_train
        self._after_iter = after_iter
        self._after_train = after_train

    def before_train(self):
        if self._before_train is not None:
            self._before_train(self.trainer)

    def after_train(self):
        if self._after_train is not None:
            self._after_train(self.trainer)

        del self._before_train
        del self._after_train
        del self._before_iter
        del self._after_iter

    def before_iter(self):
        if self._before_iter is not None:
            self._before_iter(self.trainer)

    def after_iter(self):
        if self._after_iter is not None:
            self._after_iter(self.trainer)
