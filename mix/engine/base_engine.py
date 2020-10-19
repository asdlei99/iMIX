# TODO(jinliang):jinliang_copy_and_imitate
from abc import ABCMeta, abstractmethod
import mix.engine.hooks as hooks
import weakref
from mix.engine.hooks.periods import LogBufferStorage
import logging


class EngineBase:
    """Base class for Mix engine."""

    def __init__(self):
        self._hooks = []

    def register_hooks(self, engine_hooks):
        for hk in engine_hooks:
            if hk is not None:
                assert isinstance(
                    hk, hooks.HookBase
                ), 'the current hook object must be a HookBase subclass!'
                hk.trainer = weakref.proxy(self)
                self._hooks.append(hk)

    def before_train(self):
        for hk in self._hooks:
            hk.before_train()

    def after_train(self):
        for hk in self._hooks:
            hk.after_train()

    def before_train_iter(self):
        for hk in self._hooks:
            hk.before_train_iter()

    def after_train_iter(self):
        for hk in self._hooks:
            hk.after_train_iter()
        self.log_buffer.step()

    def before_train_epoch(self):
        for hk in self._hooks:
            hk.before_train_epoch()

    def after_train_epoch(self):
        for hk in self._hooks:
            hk.after_train_epoch()

    @abstractmethod
    def run_train_epoch(self):
        pass

    @abstractmethod
    def run_train_iter(self):
        pass

    def train_by_iter(self, start_iter: int, max_iter: int) -> None:
        logger = logging.getLogger(__name__)

        self.start_iter = start_iter
        self.iter = start_iter
        self.max_iter = max_iter

        with LogBufferStorage(start_iter) as self.log_buffer:
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    self.before_train_iter()
                    self.run_train_iter()
                    self.after_train_iter()
            except Exception as e:
                raise
                # logger.error(e)
            finally:
                self.after_train()

    def train_by_epoch(self, start_epoch: int, max_epoch: int) -> None:
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.start_iter = start_epoch * len(self.data_loader)
        self.iter = self.start_iter
        self.max_iter = max_epoch * len(self.data_loader)

        with LogBufferStorage(self.iter) as self.log_buffer:
            try:
                self.before_train()
                for self.epoch in range(self.start_epoch, self.max_epoch):
                    self.before_train_epoch()
                    self.run_train_epoch()
                    self.after_train_epoch()
            except Exception as e:
                raise
            finally:
                self.after_train()
