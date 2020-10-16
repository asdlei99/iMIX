from .base_engine import EngineBase
from .organizer import Organizer
import time
import logging
import torch
import mix.utils.comm as comm
import numpy as np
from mix.evaluation import verify_results


class CommonEngine(EngineBase):

    def __init__(self, model, data_loader, optimizer, batch_processor=None):
        super(CommonEngine, self).__init__()

        model.train()

        self.model = model
        self.optimizer = optimizer
        self.batch_processor = batch_processor

        self.data_loader = data_loader
        self.__data_loader_iter = iter(data_loader)

    def run_iter(self):
        assert self.model.training, '[CommonEngine] model was changed to eval model!'
        start_time = time.perf_counter()
        batch_data = next(self.__data_loader_iter)
        data_time = time.perf_counter() - start_time

        if self.batch_processor is not None:
            self.output = self.batch_processor(
                batch_data)  # TODO(jinliang) 暂时这么处理，缺少相关参数
        else:
            self.output = self.model(batch_data)

        # TODO(jinliang) 缺少forward结果检查  -> utils模块内   -->simpleTrainer detect_anomaly
        # TODO(jinliang) 缺少backward -> optimizer 做backward
        # TODO(jinliang) 缺少forward结果保存+获取数据时间  ->logger   --> write_metrics

        self.output['loss'] /= comm.get_world_size()

        metrics_dict = self.output
        metrics_dict['data_time'] = data_time
        self._write_metrics(metrics_dict)

        # self.optimizer.zero_grad()
        # self.output['loss'].backward()
        # if self._grad_clip is not None:
        #     grad_norm = self._grad_clip(self.trainer.parameters())
        #     if grad_norm is not None:
        #         self.trainer.log_buffer.push_scalar(
        #             'grad_norm',
        #             float(grad_norm))  #TODO(jinliang) 缺少num_samples
        #
        # self.optimizer.step()

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                'Loss became infinite or NaN at iteration={}!\nloss_dict = {}'.
                format(self.iter, loss_dict))

    def _write_metrics(self, metrics_dict: dict):
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


class MixEngine(CommonEngine):

    def __init__(self, cfg):
        self.organizer = Organizer(cfg)
        super(MixEngine, self).__init__(self.organizer.model,
                                        self.organizer.train_data_loader,
                                        self.organizer.optimizer)

        self.start_iter = self.organizer.start_iter
        self.max_iter = self.organizer.max_iter
        self.cfg = self.organizer.cfg

        self.register_hooks(self.organizer.hooks)

    def train_iter(self):
        super(MixEngine, self).train_iter(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(self, '_last_eval_results'
                           ), 'No evaluation results obtained during training!'
            verify_results(self.cfg, self.organizer._last_eval_results)
            return self.organizer._last_eval_results
