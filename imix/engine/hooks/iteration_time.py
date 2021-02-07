# TODO(jinliang):jinliang_copy_and imitate
from .base_hook import HookBase
import logging
import datetime
# from imix.utils.timer import Timer
from .builder import HOOKS
from imix.utils_imix.Timer import Timer

# class IterationTimeHook(HookBase):
#     """
#     统计每次迭代时间、训练过程中总耗时及其单次平均耗时
#
#     单次跌代时间=after_train_iter-before_train_iter,但IterationTimeHook.after_train_iter必须在optimizerHook之后调用
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
#     def before_train_iter(self):
#         self._iter_start_time = time.time()
#
#     def after_train_iter(self):
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

# @HOOKS.register_module()
# class IterationTimerHook(HookBase):
#     """Track the time spent for each iteration (each run_step call in the
#     trainer). Print a summary in the end of training.
#
#     This hook uses the time between the call to its :meth:`before_step` and
#     :meth:`after_step` methods. Under the convention that :meth:`before_step`
#     of all hooks should only take negligible amount of time, the
#     :class:`IterationTimer` hook should be placed at the beginning of the list
#     of hooks to obtain accurate timing.
#     """
#
#     def __init__(self, warmup_iter=3):
#         """
#         Args:
#             warmup_iter (int): the number of iterations at the beginning to exclude
#                 from timing.
#         """
#         self._warmup_iter = warmup_iter
#         self._iter_timer = Timer()
#         self._start_time = time.perf_counter()
#         self._total_timer = Timer()
#         self._epoch_timer = Timer()
#
#     def before_train(self):
#         self._start_time = time.perf_counter()
#         self._total_timer.reset()
#         self._total_timer.pause()
#
#     def after_train(self):
#         logger = logging.getLogger(__name__)
#         total_time = time.perf_counter() - self._start_time
#         total_time_minus_hooks = self._total_timer.seconds()
#         hook_time = total_time - total_time_minus_hooks
#
#         num_iter = self.trainer.iter + 1 - self.trainer.start_iter - self._warmup_iter
#
#         if num_iter > 0 and total_time_minus_hooks > 0:
#             # Speed is meaningful only after warmup
#             # NOTE this format is parsed by grep in some scripts
#             logger.info(
#                 'Overall training speed: {} iterations in {} ({:.4f} s / it)'.
#                 format(
#                     num_iter,
#                     str(
#                         datetime.timedelta(
#                             seconds=int(total_time_minus_hooks))),
#                     total_time_minus_hooks / num_iter,
#                 ))
#
#         logger.info('Total training time: {} ({} on hooks)'.format(
#             str(datetime.timedelta(seconds=int(total_time))),
#             str(datetime.timedelta(seconds=int(hook_time))),
#         ))
#
#     def before_train_iter(self):
#         self._iter_timer.reset()
#         self._total_timer.resume()
#
#     def after_train_iter(self):
#         # +1 because we're in after_step
#         iter_done = self.trainer.iter - self.trainer.start_iter + 1
#         if iter_done >= self._warmup_iter:
#             sec = self._iter_timer.seconds()
#             self.trainer.log_buffer.put_scalars(time=sec)
#         else:
#             self._start_time = time.perf_counter()
#             self._total_timer.reset()
#
#         self._total_timer.pause()
#
#     def before_train_epoch(self):
#         self._epoch_timer.reset()
#
#     def after_train_epoch(self):
#         epoch_sec = self._epoch_timer.seconds()
#         self.trainer.log_buffer.put_scalars(epoch_time=epoch_sec)


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

  def __init__(self, warmup_iter=0):
    """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
    self._warmup_iter = warmup_iter
    self._iter_start_time = None
    self._epoch_start_time = None
    self._total_start_time = None
    self._total_end_time = None

  def before_train(self):
    self._total_start_time = Timer.now()

  def after_train(self):
    self._total_end_time = Timer.now()
    self._write_log()

  def before_train_iter(self):
    self._iter_start_time = Timer.now()

  def after_train_iter(self):
    iter_seconds = Timer.passed_seconds(
        start=self._iter_start_time, end=Timer.now())
    self.trainer.log_buffer.put_scalar('iter_time', iter_seconds)

  def before_train_epoch(self):
    self._epoch_start_time = Timer.now()

  def after_train_epoch(self):
    epoch_sec = Timer.passed_seconds(self._epoch_start_time, Timer.now())
    self.trainer.log_buffer.put_scalars(epoch_time=epoch_sec)

  def _write_log(self):
    logger = logging.getLogger(__name__)
    total_time = Timer.passed_seconds(self._total_start_time,
                                      self._total_end_time)
    num_iter = self.trainer.iter + 1 - self.trainer.start_iter - self._warmup_iter
    if num_iter > 0 and total_time > 0:
      logger.info(
          'The Whole training speed:{} iterations in {} ({:.4f} sec/iter)'
          .format(num_iter, str(datetime.timedelta(seconds=int(total_time))),
                  total_time / num_iter))
    if self.trainer.by_epoch:
      epochs = self.trainer.max_epoch
      logger.info('The Whole training speed:{} epochs in {} ({} /epoch)'.format(
          epochs, str(datetime.timedelta(seconds=int(total_time))),
          str(datetime.timedelta(seconds=int(total_time / epochs)))))

    logger.info('total training time:{}'.format(
        str(datetime.timedelta(seconds=int(total_time)))))
