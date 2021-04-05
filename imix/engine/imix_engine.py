from .base_engine import EngineBase
from .organizer import Organizer, is_mixed_precision
import logging
import torch
# import imix.utils.comm as comm
import imix.utils_imix.distributed_info as comm
import numpy as np
from imix.evaluation import verify_results
from torch.cuda.amp.autocast_mode import autocast
from imix.utils_imix.logger import setup_logger
from .hooks.periods import write_metrics
from imix.utils_imix.Timer import Timer
import time


class IterDataLoader:

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)

    def __next__(self):
        try:
            data = next(self.data_loader_iter)
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            data = next(self.data_loader_iter)
        finally:
            return data

    def __len__(self):
        return len(self.data_loader)


class CommonEngine(EngineBase):

    def __init__(self, model, data_loader, optimizer, loss_fn, batch_processor=None):
        super(CommonEngine, self).__init__()

        model.train()

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_processor = batch_processor

        self.data_loader = data_loader
        self.data_loader_iter = IterDataLoader(data_loader=data_loader)
        self.imixed_precision = False

    def run_train_iter(self):
        assert self.model.training, '[CommonEngine] model was changed to eval model!'

        # metrics_dict = {}
        # start_time = Timer.now()
        # batch_data = next(self.data_loader_iter)
        # data_time = Timer.passed_seconds(start=start_time, end=Timer.now())
        # metrics_dict['data_time'] = data_time
        loger = logging.getLogger(__name__)

        @Timer.run_time
        def load_data():
            return next(self.data_loader_iter)

        metrics_dict = {}
        batch_data, metrics_dict['data_time'] = load_data()
        # loger.info('batch_data keys:{}'.format(batch_data.keys()))

        if self.batch_processor is not None:
            self.output = self.batch_processor(batch_data)  # TODO(jinliang) 暂时这么处理，缺少相关参数
        else:
            with autocast(enabled=is_mixed_precision()):  # TODO(jinliang) autocast warp

                try:
                    self.model_output = self.model(batch_data)

                    self.output = self.loss_fn(self.model_output)

                except:
                    self.model_output = self.model(batch_data)
                    self.output = self.loss_fn(self.model_output)
                # self.output = self.loss_fn.loss(
                #     scores=self.model_output['scores'],
                #     targets=self.model_output['target'])

        # self.output['loss'] /= comm.get_world_size()
        metrics_dict.update(self.output)
        write_metrics(metrics_dict)

        # self.optimizer.zero_grad()
        # self.output['loss'].backward()
        # self.optimizer.step()

    # def _write_metrics(self,
    #                    metrics_dict: dict):  # TODO(jinliang):jinliang_copy
    #     """
    #     Args:
    #         metrics_dict (dict): dict of scalar metrics
    #     """
    #
    #     # self.log_buffer.put_scalar('total_loss', metrics_dict['loss'])
    #     # self.log_buffer.put_scalar('data_time', metrics_dict['data_time'])
    #     # return
    #
    #     import numpy as np
    #     device = metrics_dict['loss'].device
    #     with torch.cuda.stream(torch.cuda.Stream() if device.type == 'cuda' else None):
    #         metrics_dict.pop('loss')
    #         all_metrics_dict = comm.gather(metrics_dict)
    #
    #     if comm.is_main_process():
    #         data_time = np.max((m.pop('data_time') for m in all_metrics_dict))
    #         self.log_buffer.put_scalar('data_time', data_time)
    #         metrics_dict = {
    #             k:np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
    #         }

    # metrics_dict = {
    #     k: v.detach().cpu().item()
    #     if isinstance(v, torch.Tensor) else float(v)
    #     for k, v in metrics_dict.items()
    # }
    # # gather metrics among all workers for logging
    # # This assumes we do DDP-style training, which is currently the only
    # # supported method in detectron2.
    # all_metrics_dict = comm.gather(metrics_dict)
    #
    # if comm.is_main_process():
    #     if 'data_time' in all_metrics_dict[0]:
    #         # data_time among workers can have high variance. The actual latency
    #         # caused by data_time is the maximum among workers.
    #         data_time = np.max(
    #             [x.pop('data_time') for x in all_metrics_dict])
    #         self.log_buffer.put_scalar('data_time', data_time)
    #
    #     # average the rest metrics
    #     metrics_dict = {
    #         k: np.mean([x[k] for x in all_metrics_dict])
    #         for k in all_metrics_dict[0].keys()
    #     }
    #     total_losses_reduced = sum(loss for loss in metrics_dict.values())
    #
    #     self.log_buffer.put_scalar('total_loss', total_losses_reduced)
    #     if len(metrics_dict) > 1:
    #         self.log_buffer.put_scalars(**metrics_dict)

    def run_train_epoch(self):
        assert self.model.training, '[CommonEngine] model was changed to eval model!'
        time.sleep(2)  # prevent possible deadlockduring epoch transition
        for i in range(0, len(self.data_loader)):
            self.inner_iter = i
            self.before_train_iter()
            self.run_train_iter()
            self.after_train_iter()
            self.iter += 1


class imixEngine(CommonEngine):

    def __init__(self, cfg):
        self.organizer = Organizer(cfg)
        super(imixEngine, self).__init__(
            model=self.organizer.model,
            data_loader=self.organizer.train_data_loader,
            optimizer=self.organizer.optimizer,
            loss_fn=self.organizer.losses_fn)

        self.start_iter = self.organizer.start_iter
        self.max_iter = self.organizer.max_iter
        self.start_epoch = self.organizer.start_epoch
        self.max_epoch = self.organizer.max_epoch
        self.cfg = self.organizer.cfg
        self.by_epoch = self.organizer.by_epoch
        self.imixed_precision = self.organizer.imixed_precision if hasattr(self.organizer,
                                                                           'imixed_precision') else False

        self.register_hooks(self.organizer.hooks)

    def run_by_iter(self):
        super(imixEngine, self).train_by_iter(self.start_iter, self.max_iter)

    def run_by_epoch(self):
        super(imixEngine, self).train_by_epoch(self.start_epoch, self.max_epoch)

    def train(self):
        if self.by_epoch:
            self.run_by_epoch()
        else:
            self.run_by_iter()
