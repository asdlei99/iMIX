# TODO(jinliang):jinliang_copy_and_imitate
import logging
import time
import weakref
from abc import ABCMeta, abstractmethod

import imix.engine.hooks as hooks
from imix.engine.hooks.periods import LogBufferStorage


class EngineBase:
    """Base class for imix engine."""

    def __init__(self):
        self._hooks = []

    def register_hooks(self, engine_hooks):
        for hk in engine_hooks:
            if hk is not None:
                assert isinstance(hk, hooks.HookBase), 'the current hook object must be a HookBase subclass!'
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
        if self.by_epoch:
            self.log_buffer.epoch_iter(self.inner_iter)

    def before_train_epoch(self):
        for hk in self._hooks:
            hk.before_train_epoch()

    def after_train_epoch(self):
        for hk in self._hooks:
            hk.after_train_epoch()
        self.log_buffer.epoch_step()

    def before_forward(self):
        for hk in self._hooks:
            hk.before_forward()

    def after_forward(self):
        for hk in self._hooks:
            hk.after_forward()

    @abstractmethod
    def run_train_epoch(self):
        pass

    @abstractmethod
    def run_train_iter(self):
        pass

    def train_by_iter(self, start_iter: int, max_iter: int) -> None:
        self.start_iter = start_iter
        self.iter = start_iter
        self.max_iter = max_iter

        with LogBufferStorage(start_iter, by_epoch=False) as self.log_buffer:
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
                time.sleep(1)  # wait for some hooks like logger to finish
                self.after_train()

    def train_by_epoch(self, start_epoch: int, max_epoch: int) -> None:
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.start_iter = start_epoch * len(self.data_loader)
        self.iter = self.start_iter
        self.max_iter = max_epoch * len(self.data_loader)

        with LogBufferStorage(self.iter, len(self.data_loader)) as self.log_buffer:
            try:
                self.before_train()
                for self.epoch in range(self.start_epoch, self.max_epoch):
                    self.before_train_epoch()
                    self.run_train_epoch()
                    self.after_train_epoch()
            except Exception as e:
                raise
            finally:
                time.sleep(1)  # wait for some hooks like logger to finish
                self.after_train()
