import logging
import weakref
from abc import ABCMeta, abstractmethod

import mix.engine.hooks as hooks
from mix.engine.hooks.periods import LogBufferStorage


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

    def before_iter(self):
        for hk in self._hooks:
            hk.before_iter()

    def after_iter(self):
        for hk in self._hooks:
            hk.after_iter()
        self.log_buffer.step()

    @abstractmethod
    def run_iter(self):
        pass

    def train_iter(self, start_iter: int, max_iter: int) -> None:
        logger = logging.getLogger(__name__)

        self.start_iter = start_iter
        self.iter = start_iter
        self.max_iter = max_iter

        with LogBufferStorage(start_iter) as self.log_buffer:
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    self.before_iter()
                    self.run_iter()
                    self.after_iter()
            except Exception as e:
                raise
                #logger.error(e)
            finally:
                self.after_train()
