# TODO(jinliang):jinliang_imitate
from .base_hook import HookBase, PriorityStatus
from .builder import HOOKS
import imix.utils_imix.distributed_info as comm
from imix.utils_imix.file_io import PathManager
import logging
import json
import os

import torch
from operator import itemgetter
from shutil import copyfile
import copy
from imix.utils_imix.config import get_imix_work_dir
from .periods.checkpoint import CheckPointHook
from imix.utils_imix.distributed_info import master_only_run


@HOOKS.register_module()
class EvaluateHook(HookBase):
    """Run an evaluation function periodically, and at the end of training.

    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function, eval_json_file='eval_result.json'):
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
        self._level = PriorityStatus.LOW
        self._file_handle = PathManager.open(os.path.join(get_imix_work_dir(), eval_json_file), 'w')
        self._all_eval_results = []

    def _do_eval(self):
        results = self._func()
        comm.synchronize()
        return results

    def after_train_iter(self):
        if self.trainer.by_epoch is False:
            next_iter = self.trainer.iter + 1
            is_final = next_iter == self.trainer.max_iter
            if is_final or (self._period > 0 and next_iter % self._period == 0):
                results = self._do_eval()
                self._wirte_eval_result(results)
                self._write_to_tensorboard(results)

    def after_train(self):
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        if hasattr(self, '_file_handle'):
            self._file_handle.close()
        if len(self._all_eval_results):
            self._best_eval_result()
        del self._func

    def after_train_epoch(self):
        results = self._do_eval()
        self._wirte_eval_result(results)
        self._write_to_tensorboard(results, is_epoch=True)

    @master_only_run
    def _write_to_tensorboard(self, eval_result, is_epoch=False):
        for k, v in eval_result.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, dict):
                for ki, vi in v.items():
                    self.trainer.log_buffer.put_scalar(ki, vi, is_epoch=is_epoch)
            else:
                self.trainer.log_buffer.put_scalar(k, v, is_epoch=is_epoch)

    @master_only_run
    def _wirte_eval_result(self, results):

        data = self._train_info()
        data.update(self._eval_result(results))

        for hk in self.trainer._hooks:
            if isinstance(hk, CheckPointHook):
                ck_name = hk.curr_checkpoint_name
                if ck_name:
                    data.update({'model_name': ck_name})
                break

        self._all_eval_results.append(data)

        self._file_handle.write(json.dumps(data, sort_keys=False) + '\n')
        self._file_handle.flush()

        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def _train_info(self):
        train_info = {'iter': self.trainer.iter, 'max_iter': self.trainer.max_iter}
        if self.trainer.by_epoch:
            train_info.update({'epoch': self.trainer.epoch})
            train_info.update({'max_epoch': self.trainer.max_epoch})

        return train_info

    def _eval_result(self, eval_result):

        def value(v):
            return v.item() if isinstance(v, torch.Tensor) else v

        result = {k: value(v) for k, v in eval_result.items()}
        return result

    @property
    def level(self):
        return self._level

    def _best_eval_result(self):

        def get_sored_key() -> list:
            keys = list(self._all_eval_results[0].keys())
            keys.remove('max_iter')
            keys.remove('iter')
            if 'model_name' in keys:
                keys.remove('model_name')

            if self.trainer.by_epoch:
                keys.remove('epoch')
                keys.remove('max_epoch')
            return keys

        def absolute_path(file_name):
            return os.path.join(self.trainer.work_dir, file_name)

        key_sort = get_sored_key()
        results = sorted(self._all_eval_results, key=itemgetter(*key_sort), reverse=True)
        best_result = copy.deepcopy(results[0])

        best_model_name = best_result.pop('model_name')

        # best info log ouput
        if self.trainer.by_epoch:
            best_result.pop('iter')
            best_result.pop('max_iter')
            best_result.pop('max_epoch')
            best_epoch = best_result.pop('epoch')
        else:
            best_result.pop('max_iter')
            best_iter = best_result.pop('iter')
        best_info = best_epoch if self.trainer.by_epoch else best_iter
        logger = logging.getLogger(__name__)
        logger.info('In {}th epoch/iter,got the highest score{}'.format(best_info, best_result))

        # copy ck file
        copyfile(src=absolute_path(best_model_name), dst=absolute_path('best_result.pth'))
