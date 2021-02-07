# TODO(jinliang):jinliang_imitate
from .base_hook import HookBase, PriorityStatus
from .builder import HOOKS
# import imix.utils.comm as comm
import imix.utils_imix.distributed_info as comm
from imix.utils.file_io import PathManager
import logging
import json
import os
import torch


@HOOKS.register_module()
class EvaluateHook(HookBase):
  """Run an evaluation function periodically, and at the end of training.

  It is executed every ``eval_period`` iterations and after the last iteration.
  """

  def __init__(self,
               eval_period,
               eval_function,
               eval_json_file='eval_result.json'):
    """
        Args:
            eval_period (int): the period to run `eval_function`.
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
    self._period = eval_period
    self._func = eval_function
    self._level = PriorityStatus.LOWER
    self._file_handle = PathManager.open(eval_json_file, 'w')

  def _do_eval(self):
    results = self._func()
    # if results:
    #     assert isinstance(
    #         results, dict
    #     ), 'Eval function must return a dict. Got {} instead.'.format(
    #         results)
    #
    #     # flattened_results = flatten_results_dict(results)
    #     flattened_results = 111  #TODO(jinliang)
    #     for k, v in flattened_results.items():
    #         try:
    #             v = float(v)
    #         except Exception:
    #             raise ValueError(
    #                 '[EvalHook] eval_function should return a nested dict of float. '
    #                 "Got '{}: {}' instead.".format(k, v))
    #     self.trainer.log_buffer.put_scalars(
    #         **flattened_results, smoothing_hint=False)

    # Evaluation may take different time among workers.
    # A barrier make them start the next iteration together.
    comm.synchronize()
    return results

  def after_train_iter(self):
    if self.trainer.by_epoch is False:
      next_iter = self.trainer.iter + 1
      is_final = next_iter == self.trainer.max_iter
      if is_final or (self._period > 0 and next_iter % self._period == 0):
        results = self._do_eval()
        self.__wirte_eval_result(results)

  def after_train(self):
    # func is likely a closure that holds reference to the trainer
    # therefore we clean it to avoid circular reference in the end
    if hasattr(self, '_file_handle'):
      self._file_handle.close()

    del self._func

  def after_train_epoch(self):
    # logger = logging.getLogger(__name__)
    results = self._do_eval()
    self.__wirte_eval_result(results)
    # logger.info('epoch_{} evaluate accuracy :'.format(
    #     self.trainer.epoch, float(results['classification'])))

  def __wirte_eval_result(self, results):
    data = {'iter': self.trainer.iter + 1, 'max_iter': self.trainer.max_iter}
    if self.trainer.by_epoch:
      data.update({'epoch': self.trainer.epoch + 1})
      data.update({'max_epoch': self.trainer.max_epoch})

    for k, v in results.items():
      if isinstance(v, torch.Tensor):
        data[k] = v.item()
      else:
        data[k] = v

    self._file_handle.write(json.dumps(data, sort_keys=False) + '\n')
    self._file_handle.flush()

    try:
      os.fsync(self._file_handle.fileno())
    except AttributeError:
      pass

  @property
  def level(self):
    return self._level
