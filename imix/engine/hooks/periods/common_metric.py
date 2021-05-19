# TODO(jinliang):jinliang_copy
import datetime
import logging
from typing import NamedTuple, Optional

import torch

from imix.utils_imix.Timer import Timer
from .log_buffer_imix import LogBufferWriter, get_log_buffer

# from imix.utils.logger import setup_logger

# class CommonMetricLoggerHook(LogBufferWriter):
#     """在终端输出相关信息，迭代时间、前传相关Loss、学习率、ETA."""
#
#     def __init__(self, max_iter):
#         self._max_iter = max_iter
#         self.logger = logging.getLogger(__name__)
#         self._recorder_iter_time = None
#
#     # def write(self):
#     #     """
#     #     获取LogBufferStorage存储的数据，根据需求拿到当前迭代次数、加载数据所需时间、学习率、CUDA剩余内存等，并将这些数据存入logger中
#     #     :return:
#     #     """
#     #     pass
#
#     def write(self):  # TODO(jinliang):modify
#         #self.logger.info("get_world_size{}  get_local_rank{} get_rank{}".format(
#           comm.get_world_size(),comm.get_local_rank(),comm.get_rank()))
#         storage = get_log_buffer()
#         iteration = storage.iter
#
#         try:
#             data_time = storage.history('data_time').avg(20)
#         except KeyError:
#             # they may not exist in the first few iterations (due to warmup)
#             # or when SimpleTrainer is not used
#             data_time = None
#
#         eta_string = None
#         try:
#             iter_time = storage.history('time').global_avg()
#             eta_seconds = storage.history('time').median(1000) * (
#                 self._max_iter - iteration)
#             storage.put_scalar(
#                 'eta_seconds', eta_seconds, smoothing_hint=False)
#             eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
#         except KeyError:
#             iter_time = None
#             # estimate eta on our own - more noisy
#             if self._recorder_iter_time is not None:
#                 estimate_iter_time = (time.perf_counter() -
#                                       self._recorder_iter_time[1]) / (
#                                           iteration - self._recorder_iter_time[0])
#                 eta_seconds = estimate_iter_time * (self._max_iter - iteration)
#                 eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
#             self._recorder_iter_time = (iteration, time.perf_counter())
#
#         try:
#             lr = '{:.6f}'.format(storage.history('lr').latest())
#         except KeyError:
#             lr = 'N/A'
#
#         if torch.cuda.is_available():
#             max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
#         else:
#             max_mem_mb = None
#
#         if storage.by_epoch:
#             # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
#             self.logger.info(
#                 ' {eta}  epoch:{epoch}   {inner_iter}/{single_epoch_iters}   iter: {iter}
#                   max_iter:{max_iter}  {losses}  {time}{data_time}lr: {lr}  {memory}'
#                 .format(
#                     eta=f'eta: {eta_string}  ' if eta_string else '',
#                     epoch=storage.epoch,
#                     inner_iter=storage.epoch_inner_iter + 1,
#                     single_epoch_iters=storage.single_epoch_iters,
#                     iter=iteration + 1,
#                     max_iter=self._max_iter,
#                     losses='  '.join([
#                         '{}: {:.3f}'.format(k, v.median(20))
#                         for k, v in storage.histories().items() if 'loss' in k
#                     ]),
#                     time='time: {:.4f}  '.format(iter_time)
#                     if iter_time is not None else '',
#                     data_time='data_time: {:.4f}  '.format(data_time)
#                     if data_time is not None else '',
#                     lr=lr,
#                     memory='max_mem: {:.0f}M'.format(max_mem_mb)
#                     if max_mem_mb is not None else '',
#                 ))
#         else:
#             self.logger.info(
#                 ' {eta} iter: {iter} max_iter:{max_iter}  {losses}  {time}{data_time}lr: {lr}  {memory}'
#                 .format(
#                     eta=f'eta: {eta_string}  ' if eta_string else '',
#                     iter=iteration + 1,
#                     max_iter=self._max_iter,
#                     losses='  '.join([
#                         '{}: {:.3f}'.format(k, v.median(20))
#                         for k, v in storage.histories().items() if 'loss' in k
#                     ]),
#                     time='time: {:.4f}  '.format(iter_time)
#                     if iter_time is not None else '',
#                     data_time='data_time: {:.4f}  '.format(data_time)
#                     if data_time is not None else '',
#                     lr=lr,
#                     memory='max_mem: {:.0f}M'.format(max_mem_mb)
#                     if max_mem_mb is not None else '',
#                 ))

_RecorderTime = NamedTuple(
    '_RecorderTime',
    [('iteration', int), ('time', float)],
)


class CommonMetricLoggerHook(LogBufferWriter):
    """在终端输出相关信息，迭代时间、前传相关Loss、学习率、ETA."""

    def __init__(self, max_iter: int, max_epoch: Optional[int] = None):
        self._max_iter = max_iter
        self._max_epoch = max_epoch
        self.logger = logging.getLogger(__name__)
        self._recorder_iter_time: _RecorderTime = None

    def process_buffer_data(self):
        if self.log_buffer.by_epoch:
            # self._epoch_metric()
            self.__epoch_metric()
        else:
            # self._iter_metric()
            self.__iter_metric()

    # def _epoch_metric(self):
    #     storage = self.log_buffer
    #     iteration = storage.iter
    #
    #     try:
    #         data_time = storage.history('data_time').avg(20)
    #     except KeyError:
    #         # they may not exist in the first few iterations (due to warmup)
    #         # or when SimpleTrainer is not used
    #         data_time = None
    #
    #     eta_string = None
    #     try:
    #         iter_time = storage.history('time').global_avg()
    #         eta_seconds = storage.history('time').median(1000) * (
    #                 self._max_iter - iteration)
    #         storage.put_scalar(
    #             'eta_seconds', eta_seconds, smoothing_hint=False)
    #         eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    #     except KeyError:
    #         iter_time = None
    #         # estimate eta on our own - more noisy
    #         if self._recorder_iter_time is not None:
    #             estimate_iter_time = (time.perf_counter() -
    #                                   self._recorder_iter_time[1]) / (
    #                                          iteration - self._recorder_iter_time[0])
    #             eta_seconds = estimate_iter_time * (self._max_iter - iteration)
    #             eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    #         self._recorder_iter_time = (iteration, time.perf_counter())
    #
    #     try:
    #         lr = '{:.6f}'.format(storage.history('lr').latest())
    #     except KeyError:
    #         lr = 'N/A'
    #
    #     if torch.cuda.is_available():
    #         max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
    #     else:
    #         max_mem_mb = None
    #
    #     self.logger.info(
    #         ' {eta}  epoch:{epoch}   {inner_iter}/{single_epoch_iters}
    #           iter: {iter}  max_iter:{max_iter}  {losses}  {time}{data_time}lr: {lr}  {memory}'
    #             .format(
    #             eta=f'eta: {eta_string}  ' if eta_string else '',
    #             epoch=storage.epoch,
    #             inner_iter=storage.epoch_inner_iter + 1,
    #             single_epoch_iters=storage.single_epoch_iters,
    #             iter=iteration + 1,
    #             max_iter=self._max_iter,
    #             losses='  '.join([
    #                 '{}: {:.3f}'.format(k, v.median(20))
    #                 for k, v in storage.histories().items() if 'loss' in k
    #             ]),
    #             time='time: {:.4f}  '.format(iter_time)
    #             if iter_time is not None else '',
    #             data_time='data_time: {:.4f}  '.format(data_time)
    #             if data_time is not None else '',
    #             lr=lr,
    #             memory='max_mem: {:.0f}M'.format(max_mem_mb)
    #             if max_mem_mb is not None else '',
    #         ))

    # def _iter_metric(self):
    #     storage = self.log_buffer
    #     iteration = storage.iter
    #
    #     try:
    #         data_time = storage.history('data_time').avg(20)
    #     except KeyError:
    #         # they may not exist in the first few iterations (due to warmup)
    #         # or when SimpleTrainer is not used
    #         data_time = None
    #
    #     eta_string = None
    #     try:
    #         iter_time = storage.history('time').global_avg()
    #         eta_seconds = storage.history('time').median(1000) * (
    #                 self._max_iter - iteration)
    #         storage.put_scalar(
    #             'eta_seconds', eta_seconds, smoothing_hint=False)
    #         eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    #     except KeyError:
    #         iter_time = None
    #         # estimate eta on our own - more noisy
    #         if self._recorder_iter_time is not None:
    #             estimate_iter_time = (time.perf_counter() -
    #                                   self._recorder_iter_time[1]) / (
    #                                          iteration - self._recorder_iter_time[0])
    #             eta_seconds = estimate_iter_time * (self._max_iter - iteration)
    #             eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    #         self._recorder_iter_time = (iteration, time.perf_counter())
    #
    #     iter_time = None
    #     # estimate eta on our own - more noisy
    #     if self._recorder_iter_time is not None:
    #         estimate_iter_time = (time.perf_counter() -
    #                               self._recorder_iter_time[1]) / (
    #                                      iteration - self._recorder_iter_time[0])
    #         eta_seconds = estimate_iter_time * (self._max_iter - iteration)
    #         eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    #     self._recorder_iter_time = (iteration, time.perf_counter())
    #
    #     try:
    #         lr = '{:.6f}'.format(storage.history('lr').latest())
    #     except KeyError:
    #         lr = 'N/A'
    #
    #     if torch.cuda.is_available():
    #         max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
    #     else:
    #         max_mem_mb = None
    #
    #     self.logger.info(
    #         ' {eta} iter: {iter} max_iter:{max_iter}  {losses}  {time}{data_time}lr: {lr}  {memory}'
    #             .format(
    #             eta=f'eta: {eta_string}  ' if eta_string else '',
    #             iter=iteration + 1,
    #             max_iter=self._max_iter,
    #             losses='  '.join([
    #                 '{}: {:.3f}'.format(k, v.median(20))
    #                 for k, v in storage.histories().items() if 'loss' in k
    #             ]),
    #             time='time: {:.4f}  '.format(iter_time)
    #             if iter_time is not None else '',
    #             data_time='data_time: {:.4f}  '.format(data_time)
    #             if data_time is not None else '',
    #             lr=lr,
    #             memory='max_mem: {:.0f}M'.format(max_mem_mb)
    #             if max_mem_mb is not None else '',
    #         ))

    @staticmethod
    def __get_used_max_memory():
        if torch.cuda.is_available():
            max_used_memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            return 'max used memory:{:.0f}M'.format(max_used_memory)
        else:
            return ''

    def __get_lr(self):
        try:
            lr = '{:.6e}'.format(self.log_buffer.history('lr').latest())
        except KeyError:
            lr = 'N/A'
        finally:
            return lr

    def __get_data_time(self):
        try:
            data_time = self.log_buffer.history('data_time').avg(20)
        except KeyError:
            data_time = None
        finally:
            if data_time is not None:
                return '{:.4f}'.format(data_time)
            else:
                return ''

    def __get_by_iter_train_time(self):
        try:
            iter_time = self.log_buffer.history('iter_time').global_mean
            eta_time = self.log_buffer.history('iter_time').median(1000) * (self._max_iter - self.log_buffer.iter)
            eta_time_str = str(datetime.timedelta(seconds=int(eta_time)))
            iter_time_str = str(iter_time)
        except KeyError:
            eta_time_str = ''
            iter_time_str = ''
            if self._recorder_iter_time is not None:
                estimate_iter_time = (Timer.now() - self._recorder_iter_time.time) / (
                    self.log_buffer.iter - self._recorder_iter_time.iteration)
                eta_time = estimate_iter_time * (self._max_iter - self.log_buffer.iter)
                eta_time_str = str(datetime.timedelta(seconds=int(eta_time)))
                iter_time_str = str(datetime.timedelta(seconds=int(estimate_iter_time)))

            self._recorder_iter_time = _RecorderTime(self.log_buffer.iter, Timer.now())
        finally:
            return iter_time_str, eta_time_str

    def __collect_common_log_info(self):
        data_time = self.__get_data_time()
        iter_time, eta_time = self.__get_by_iter_train_time()
        lr = self.__get_lr()
        max_used_memory = self.__get_used_max_memory()

        common_log_info = '{losses} \t lr:{lr} \t' \
                          'load_data_time:{data_time} \t iter_time:{iter_time} \t ' \
                          '{memory} \t eta:{eta}'.format(losses=self.__get_losses_log_info(), lr=lr,
                                                         data_time=data_time,
                                                         iter_time=iter_time,
                                                         memory=max_used_memory,
                                                         eta=eta_time
                                                         )
        return common_log_info

    def __get_losses_log_info(self):
        losses_info = []
        for k, v in self.log_buffer.histories().items():
            if 'loss' in k:
                # loss_str = '{}: {:.3f}'.format(k, v.median(20))
                loss_str = '{}: {:.3f}'.format(k, v.latest())
                losses_info.append(loss_str)

        return '  '.join(losses_info)

    def __epoch_metric(self):
        epoch = self.log_buffer.epoch
        iteration = self.log_buffer.iter + 1
        single_epoch_iters = self.log_buffer.single_epoch_iters
        epoch_inner_iter = self.log_buffer.iter % single_epoch_iters + 1
        epoch_log_info_0 = 'epoch:{epoch} \t {inner_iter}/{single_epoch_iters} \t ' \
                           'current_iter:{iter} \t ' \
                           'max_iter:{max_iter} \t ' \
                           'max_epoch:{max_epoch} \t '.format(epoch=epoch,
                                                              inner_iter=epoch_inner_iter,
                                                              single_epoch_iters=single_epoch_iters,
                                                              iter=iteration,
                                                              max_iter=self._max_iter,
                                                              max_epoch=self._max_epoch)

        epoch_log_info_1 = self.__collect_common_log_info()
        self.logger.info(epoch_log_info_0 + epoch_log_info_1)

    def __iter_metric(self):
        iteration = self.log_buffer.iter + 1
        epoch_log_info_0 = 'current_iter:{iter} \t max_iter:{max_iter} \t '.format(
            iter=iteration, max_iter=self._max_iter)

        epoch_log_info_1 = self.__collect_common_log_info()
        self.logger.info(epoch_log_info_0 + epoch_log_info_1)


def write_metrics(data: dict):
    if 'loss' in data.keys():
        data.pop('loss')

    logger_buffer = get_log_buffer()
    for k, v in data.items():
        logger_buffer.put_scalar(k, v)
