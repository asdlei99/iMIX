from .base_engine import EngineBase
from .organizer import Organizer, is_mixed_precision
import time
import logging
import torch
# import mix.utils.comm as comm
import mix.utils_mix.distributed_info as comm
import numpy as np
from mix.evaluation import verify_results
from torch.cuda.amp.autocast_mode import autocast
from mix.utils_mix.logger import setup_logger


class CommonEngine(EngineBase):

    def __init__(self,
                 model,
                 data_loader,
                 optimizer,
                 loss_fn,
                 batch_processor=None):
        super(CommonEngine, self).__init__()

        model.train()

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_processor = batch_processor

        self.data_loader = data_loader
        self.__data_loader_iter = iter(data_loader)
        self.mixed_precision = False

    def run_train_iter(self, batch_data=None):
        assert self.model.training, '[CommonEngine] model was changed to eval model!'

        metrics_dict = {}
        if (self.by_epoch is False) and (batch_data is None):
            start_time = time.perf_counter()
            batch_data = next(self.__data_loader_iter)
            data_time = time.perf_counter() - start_time
            metrics_dict['data_time'] = data_time
        else:
            metrics_dict['data_time'] = 0

        if self.batch_processor is not None:
            self.output = self.batch_processor(
                batch_data)  # TODO(jinliang) 暂时这么处理，缺少相关参数
        else:
            with autocast(enabled=is_mixed_precision()
                          ):  # TODO(jinliang) autocast warp
                self.model_output = self.model(batch_data)
                self.output = self.loss_fn(
                    dict(scores=self.model_output['scores']),
                    self.model_output['target']
                )  # TODO(jinliang) -> loss_fn(output,target)

        self.output['loss'] /= comm.get_world_size()
        metrics_dict.update(self.output)
        self._write_metrics(metrics_dict)

        # self.optimizer.zero_grad()
        # self.output['loss'].backward()
        # self.optimizer.step()

    def _detect_anomaly(self, losses,
                        loss_dict):  # TODO(jinliang):jinliang_copy
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                'Loss became infinite or NaN at iteration={}!\nloss_dict = {}'.
                format(self.iter, loss_dict))

    def _write_metrics(self,
                       metrics_dict: dict):  # TODO(jinliang):jinliang_copy
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """

        self.log_buffer.put_scalar('total_loss', metrics_dict['loss'])
        self.log_buffer.put_scalar('data_time', metrics_dict['data_time'])
        return

        metrics_dict = {
            k: v.detach().cpu().item()
            if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if 'data_time' in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max(
                    [x.pop('data_time') for x in all_metrics_dict])
                self.log_buffer.put_scalar('data_time', data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.log_buffer.put_scalar('total_loss', total_losses_reduced)
            if len(metrics_dict) > 1:
                self.log_buffer.put_scalars(**metrics_dict)

    def run_train_epoch(self):
        assert self.model.training, '[CommonEngine] model was changed to eval model!'
        time.sleep(2)  # prevent possible deadlockduring epoch transition
        for i, batch_data in enumerate(self.data_loader):
            self.inner_iter = i
            self.before_train_iter()
            self.run_train_iter(batch_data)
            self.after_train_iter()
            self.iter += 1


class MixEngine(CommonEngine):

    def __init__(self, cfg):
        self.organizer = Organizer(cfg)
        loss_fn = self.organizer.model.module.losses if comm.get_world_size(
        ) > 1 else self.organizer.model.losses
        super(MixEngine, self).__init__(
            self.organizer.model,
            self.organizer.train_data_loader,
            self.organizer.optimizer,
            loss_fn=loss_fn)

        self.start_iter = self.organizer.start_iter
        self.max_iter = self.organizer.max_iter
        self.start_epoch = self.organizer.start_epoch
        self.max_epoch = self.organizer.max_epoch
        self.cfg = self.organizer.cfg
        self.by_epoch = self.organizer.by_epoch
        self.mixed_precision = self.organizer.mixed_precision if hasattr(
            self.organizer, 'mixed_precision') else False

        self.register_hooks(self.organizer.hooks)

    def train_by_iter(self):
        super(MixEngine, self).train_by_iter(self.start_iter, self.max_iter)
        # if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
        #     assert hasattr(self, '_last_eval_results'
        #                    ), 'No evaluation results obtained during training!'
        #     verify_results(self.cfg, self.organizer._last_eval_results)
        #     return self.organizer._last_eval_results

    def train_by_epoch(self):
        super(MixEngine, self).train_by_epoch(self.start_epoch, self.max_epoch)

    def train(self):
        if self.by_epoch:
            self.train_by_epoch()
        else:
            self.train_by_iter()
