import datetime
import logging
import time

from mix.utils.timer import Timer
from .base_hook import HookBase
from .builder import HOOKS

# class IterationTimeHook(HookBase):
#     """
#     统计每次迭代时间、训练过程中总耗时及其单次平均耗时
#
#     单次跌代时间=after_iter-before_iter,但IterationTimeHook.after_iter必须在optimizerHook之后调用
#
#     """
#
#     def __init__(self, have_warmup_iter=True, warmup_iter=4):
#         self._have_warmup_iter = have_warmup_iter
#         self._warmup_iter = warmup_iter if have_warmup_iter else 0
#         self._iter_start_time = None
#         self._total_start_time = None
#
#     def before_train(self):
#         self._total_start_time = time.time()
#
#     def before_iter(self):
#         self._iter_start_time = time.time()
#
#     def after_iter(self):
#         iter_num = self.trainer.iter - self.trainer.start_iter - 1
#         if iter_num > self._warmup_iter:
#             self.trainer.log_buffer.put_scalars(iter_time=time.time() -
#                                                 self._start_time)
#
#     def after_train(self):
#         total_train_time = time.time() - self._total_start_time
#         total_iter_num = self.trainer.iter - self.trainer.start_iter + 1
#
#         logger = logging.getLogger(__name__)
#         if total_iter_num > 0 and total_train_time > 0:
#             logger.info(
#                 "Overall training speed: {} iteration in {} ({:.4f} s/ it)".
#                 format(total_iter_num,
#                        str(datetime.timedelta(seconds=int(total_train_time))),
#                        total_train_time / total_iter_num))
#
#         logger.info("Total training time:{}".format(
#             str(datetime.timedelta(seconds=total_train_time))))


@HOOKS.register_module()
class IterationTimerHook(HookBase):
    """Track the time spent for each iteration (each run_step call in the
    trainer). Print a summary in the end of training.

    This hook uses the time between the call to its :meth:`before_step` and
    :meth:`after_step` methods. Under the convention that :meth:`before_step`
    of all hooks should only take negligible amount of time, the
    :class:`IterationTimer` hook should be placed at the beginning of the list
    of hooks to obtain accurate timing.
    """

    def __init__(self, warmup_iter=3):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()
        self._start_time = time.perf_counter()
        self._total_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer.reset()
        self._total_timer.pause()

    def after_train(self):
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                'Overall training speed: {} iterations in {} ({:.4f} s / it)'.
                format(
                    num_iter,
                    str(
                        datetime.timedelta(
                            seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                ))

        logger.info('Total training time: {} ({} on hooks)'.format(
            str(datetime.timedelta(seconds=int(total_time))),
            str(datetime.timedelta(seconds=int(hook_time))),
        ))

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        # +1 because we're in after_step
        iter_done = self.trainer.iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.trainer.log_buffer.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()
