# TODO(jinliang):jinliang_imitate
from .base_hook import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class CallBackHook(HookBase):

    def __init__(self,
                 *,
                 before_train=None,
                 after_train=None,
                 before_train_iter=None,
                 after_train_iter=None,
                 before_train_epoch=None,
                 after_train_epoch=None):
        self._before_train_iter = before_train_iter
        self._before_train = before_train
        self._after_train_iter = after_train_iter
        self._after_train = after_train
        self._before_train_epoch = before_train_epoch
        self._after_train_epoch = after_train_epoch

    def before_train(self):
        if self._before_train is not None:
            self._before_train(self.trainer)

    def after_train(self):
        if self._after_train is not None:
            self._after_train(self.trainer)

        del self._before_train
        del self._after_train
        del self._before_train_iter
        del self._after_train_iter
        del self._before_train_epoch
        del self._after_train_epoch

    def before_train_iter(self):
        if self._before_train_iter is not None:
            self._before_train_iter(self.trainer)

    def after_train_iter(self):
        if self._after_train_iter is not None:
            self._after_train_iter(self.trainer)

    def before_train_epoch(self):
        if self._before_train_epoch is not None:
            self._before_train_epoch(self.trainer)

    def after_train_epoch(self):
        if self._after_train_epoch is not None:
            self.after_train_epoch(self.trainer)
