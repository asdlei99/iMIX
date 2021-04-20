# TODO(jinliang):jinliang_copy_and_imitate
from .base_hook import HookBase, PriorityStatus
from .builder import HOOKS
from .periods import LogBufferWriter
from .periods.tensorboard_logger import TensorboardLoggerHook


@HOOKS.register_module()
class PeriodicLogger(HookBase):
    """每间隔一段时间，去记录一次数据.

    训练每执行一定次数后，去记录一次数据，记录分为两种形式:iter和epoch iter间隔次数和epoch间隔次数通过cfg文件获取，
    """

    def __init__(self, loggers, log_config_period):
        for logger in loggers:
            assert isinstance(logger, LogBufferWriter), logger
        self._loggers = loggers
        self._period_iter = log_config_period
        self._level = PriorityStatus.LOWER

    def before_train(self):  # TODO(jinliang) delete?
        # TODO(jinliang) 通过self.trainer.cfg(EngineBase)获取iter间隔次数和epoch间隔次数
        self._period_iter = self.trainer.cfg.log_config.period

    def after_train_iter(self):
        assert self._period_iter, self._period_iter
        write_flag = False
        if self.trainer.by_epoch is True:
            if (self.trainer.inner_iter + 1) % self._period_iter == 0:
                write_flag = True
        else:
            if (self.trainer.iter + 1) % self._period_iter == 0 or (self.trainer.iter == self.trainer.max_iter - 1):
                write_flag = True
        if write_flag:
            for logger in self._loggers:
                logger.write()

    def after_train(self):
        for logger in self._loggers:
            logger.close()

    def after_train_epoch(self):  # TODO(jinliang): modify:write epoch log
        for logger in self._loggers:
            if isinstance(logger, TensorboardLoggerHook):
                logger.write()

        # if self.trainer.epoch == self.trainer.max_epoch - 1:
        #     for logger in self._loggers:
        #         logger.write()

    @property
    def level(self):
        return self._level
