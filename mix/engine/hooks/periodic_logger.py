from .base_hook import HookBase
# from mix.utils.events import EventWriter
from .periods import LogBufferWriter
from .builder import HOOKS


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

    def before_train(self):  #TODO(jinliang) delete?
        # TODO(jinliang) 通过self.trainer.cfg(EngineBase)获取iter间隔次数和epoch间隔次数
        self._period_iter = self.trainer.cfg.log_config.period

    def after_iter(self):
        assert self._period_iter, self._period_iter
        if (self.trainer.iter + 1) % self._period_iter == 0 or (
                self.trainer.iter == self.trainer.max_iter - 1):
            for logger in self._loggers:
                logger.write()

    def after_train(self):
        for logger in self._loggers:
            logger.close()
