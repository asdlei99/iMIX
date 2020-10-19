# TODO(jinliang):jinliang_copy
from .log_buffer import LogBufferWriter, get_log_buffer
import logging
import datetime
import time
import torch


class CommonMetricLoggerHook(LogBufferWriter):
    """在终端输出相关信息，迭代时间、前传相关Loss、学习率、ETA."""

    def __init__(self, max_iter):
        self._max_iter = max_iter
        self.logger = logging.getLogger(__name__)
        self._last_write = None

    # def write(self):
    #     """
    #     获取LogBufferStorage存储的数据，根据需求拿到当前迭代次数、加载数据所需时间、学习率、CUDA剩余内存等，并将这些数据存入logger中
    #     :return:
    #     """
    #     pass

    def write(self):  # TODO(jinliang):modify
        storage = get_log_buffer()
        iteration = storage.iter

        try:
            data_time = storage.history('data_time').avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None

        eta_string = None
        try:
            iter_time = storage.history('time').global_avg()
            eta_seconds = storage.history('time').median(1000) * (
                self._max_iter - iteration)
            storage.put_scalar(
                'eta_seconds', eta_seconds, smoothing_hint=False)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            iter_time = None
            # estimate eta on our own - more noisy
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() -
                                      self._last_write[1]) / (
                                          iteration - self._last_write[0])
                eta_seconds = estimate_iter_time * (self._max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())

        try:
            lr = '{:.6f}'.format(storage.history('lr').latest())
        except KeyError:
            lr = 'N/A'

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            ' {eta}iter: {iter}  {losses}  {time}{data_time}lr: {lr}  {memory}'
            .format(
                eta=f'eta: {eta_string}  ' if eta_string else '',
                iter=iteration,
                losses='  '.join([
                    '{}: {:.3f}'.format(k, v.median(20))
                    for k, v in storage.histories().items() if 'loss' in k
                ]),
                time='time: {:.4f}  '.format(iter_time)
                if iter_time is not None else '',
                data_time='data_time: {:.4f}  '.format(data_time)
                if data_time is not None else '',
                lr=lr,
                memory='max_mem: {:.0f}M'.format(max_mem_mb)
                if max_mem_mb is not None else '',
            ))
