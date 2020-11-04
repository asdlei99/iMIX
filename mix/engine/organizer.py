# TODO(jinliang):jinliang_imitate

from ..data import build_mix_test_loader, build_mix_train_loader
from mix.utils.logger import setup_logger
from mix.utils.mix_checkpoint import MixCheckpointer
from mix.models import build_model
from mix.solver import build_lr_scheduler, build_optimizer
import mix.engine.hooks as hooks
from mix.utils.precise_bn import get_bn_modules
import mix.utils.comm as comm
from mix.evaluation import DatasetEvaluator, inference_on_dataset, build_submit_file, build_test_predict_result
from collections import OrderedDict
from mix.evaluation import VQAEvaluator
from torch.nn.parallel import DistributedDataParallel
import torch

import logging
import os

_AUTOMATIC_MIXED_PRECISION = False


def is_mixed_precision():
    return _AUTOMATIC_MIXED_PRECISION


def get_masked_fill_value():
    if is_mixed_precision():
        return torch.finfo(torch.float16).min
    else:
        return -1e9


def is_multi_gpus_mixed_precision():
    if comm.get_world_size() > 1 and is_mixed_precision():
        return True
    else:
        return False


class Organizer:

    def __init__(self, cfg):
        assert cfg, 'cfg must be non-empty!'

        logger = logging.getLogger('MIX')
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        self.cfg = cfg

        self.model = self.build_model(cfg)

        self.train_data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            # self.model = DistributedDataParallel(
            #     self.model,
            #     device_ids=[comm.get_local_rank()],
            #     broadcast_buffers=False,
            #     find_unused_parameters=True)
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[comm.get_local_rank()],
                output_device=comm.get_local_rank(),
                check_reduction=True,
                broadcast_buffers=True,
                find_unused_parameters=cfg.find_unused_parameters)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        self.checkpointer = MixCheckpointer(
            self.model,
            cfg.work_dir,
            optimizer=self.optimizer,
            scheduler=self.scheduler)

        self._by_epoch = False if hasattr(cfg, 'by_iter') else True
        self.start_epoch = 0
        self.start_iter = 0
        self.max_iter = cfg.total_epochs * len(
            self.train_data_loader) if self.by_epoch else cfg.max_iter
        self.max_epoch = cfg.total_epochs if self.by_epoch else 0

        self.hooks = self.build_hooks()

        logger.info('Organizer.init')

    @classmethod
    def build_model(cls, cfg):

        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info('build model: '.format(model))

        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg.optimizer, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg.lr_config, optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_mix_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_mix_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """TODO(jinliang) if there are many types of evaluator,it will be
        written in the form of dict or hook later,and directly init."""
        if output_folder is None:
            output_folder = os.path.join(cfg.work_dir, 'inference')

        evaluator_type = cfg.evaluator_type
        if evaluator_type == 'VQA':
            return VQAEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):  # TODO(jinliang)
        '''
        MIX框架hook初始化流程：
        1. 默认hook配置文件——默认且必须
        2. cfg设置的hook
        3. 1+2 --> 整合  --> 初始化

        整合位置：init_set()

        初始化步骤：  --> hooks.build_hooks(整合之后的Hooks)
        1. 列表遍历每个hook
        2. 根据Hook类型，调用对应构造函数完成初始化  build_from_cfg(cfg,HOOKS)
        3. 添加到self.hooks中，添加时根据优先级确定其先后顺序
        '''
        cfg = self.cfg

        hook_list = []
        if hasattr(self.cfg, 'fp16'):
            hook_list.append(
                hooks.Fp16OptimizerHook(self.cfg.optimizer_config.grad_clip,
                                        self.cfg.fp16))
            self.set_mixed_precision(True)
        else:
            hook_list.append(
                hooks.OptimizerHook(self.cfg.optimizer_config.grad_clip))
        hook_list.append(hooks.IterationTimerHook())
        hook_list.append(hooks.LRSchedulerHook(self.optimizer, self.scheduler))

        if hasattr(cfg, 'test') and hasattr(cfg.test, 'precise_bn'):
            if cfg.test.precise_bn and get_bn_modules(self.model):
                hook_list.append(
                    hooks.PreciseBNHook(cfg.test.eval_period, self.model,
                                        self.build_train_loader(cfg),
                                        cfg.test.precise_bn.num_iter))
        if comm.is_main_process():
            hook_list.append(
                hooks.CheckPointHook(self.checkpointer,
                                     cfg.checkpoint_config.period))

        if hasattr(cfg, 'test_data') and cfg.test_data.eval_period is not 0:
            hook_list.append(self.add_evaluate_hook())

        if comm.is_main_process():
            hook_list.append(
                hooks.PeriodicLogger(self.build_writers(),
                                     cfg.log_config.period))

        return hook_list

    def build_writers(self):  # TODO(jinliang) Modify based on cfg file
        return [
            hooks.CommonMetricLoggerHook(self.max_iter),
            hooks.JSONLoggerHook(
                os.path.join(self.cfg.work_dir, 'metrics.json')),
            hooks.TensorboardXLoggerHook(self.cfg.work_dir)
        ]

    def __getattr__(self, item):
        return self.hooks

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
                Args:
                    cfg (CfgNode):
                    model (nn.Module):
                    evaluators (list[DatasetEvaluator] or None): if None, will call
                        :meth:`build_evaluator`. Otherwise, must have the same length as
                        `cfg.DATASETS.TEST`.

                Returns:
                    dict: a dict of result metrics
                """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(
                cfg.DATASETS.TEST) == len(evaluators), '{} != {}'.format(
                    len(cfg.DATASETS.TEST), len(evaluators))

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.test_datasets):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warning(
                        'No evaluator found. Use `DefaultTrainer.test(evaluators=)`, '
                        'or implement its `build_evaluator` method.')
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), 'Evaluator must return a dict on the main process. Got {} instead.'.format(
                    results_i)
                logger.info('Evaluation results for {} in csv format:'.format(
                    dataset_name))
                # print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]

        logger.info('test finish')
        return results

    @classmethod
    def build_test_result(cls,
                          cfg,
                          model,
                          evaluators=None,
                          only_test_pred=False):
        """
                Args:
                    cfg (CfgNode):
                    model (nn.Module):
                    evaluators (list[DatasetEvaluator] or None): if None, will call
                        :meth:`build_evaluator`. Otherwise, must have the same length as
                        `cfg.DATASETS.TEST`.

                Returns:
                    dict: a dict of result metrics
                """
        logger = logging.getLogger(__name__)
        logger.info('build  submission result')

        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(
                cfg.DATASETS.TEST) == len(evaluators), '{} != {}'.format(
                    len(cfg.DATASETS.TEST), len(evaluators))

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.test_datasets):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warning(
                        'No evaluator found. Use `DefaultTrainer.test(evaluators=)`, '
                        'or implement its `build_evaluator` method.')
                    results[dataset_name] = {}
                    continue
            if only_test_pred:
                build_test_predict_result(model, data_loader, evaluator)
            else:
                build_submit_file(model, data_loader, evaluator)

    def add_evaluate_hook(self):

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        return hooks.EvaluateHook(self.cfg.test_data.eval_period,
                                  test_and_save_results)

    @property
    def by_epoch(self):
        return self._by_epoch

    def set_mixed_precision(self, enable=False):
        global _AUTOMATIC_MIXED_PRECISION
        _AUTOMATIC_MIXED_PRECISION = enable
