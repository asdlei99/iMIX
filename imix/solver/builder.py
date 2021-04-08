import copy
import inspect
import logging

import torch

from ..utils_imix.registry import Registry, build_from_cfg
# import imix.utils.comm as comm
# import imix.utils_imix.distributed_info as comm
OPTIMIZERS = Registry('optimizer')
OPTIMIZER_BUILDERS = Registry('optimizer builder')
LR_SCHEDULERS = Registry('lr scheduler')


def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()


def build_optimizer_constructor(cfg):
    return build_from_cfg(cfg, OPTIMIZER_BUILDERS)


def build_optimizer(optimizer_config, model):
    optimizer_cfg = copy.deepcopy(optimizer_config)
    constructor_type = optimizer_cfg.pop('constructor', 'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(type=constructor_type, optimizer_cfg=optimizer_cfg, paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer


def build_lr_scheduler(lr_config, optimizer):  # TODO(jinliang): The config file of LR does not match the code

    logger = logging.getLogger(__name__)
    try:
        assert 'policy' in lr_config
        policy_type = lr_config.pop('policy')
        # lr_type = policy_type + ''
        lr_config['type'] = policy_type
        lr_config['optimizer'] = optimizer
        # lr_config = modify_lr_config(lr_config, optimizer)#TODO(jinliang):modify

        lr_scheduler = build_from_cfg(lr_config, LR_SCHEDULERS)
    except AssertionError:
        logger.error('policy is not in {}'.format(lr_config))
    except Exception as e:  # TODO(jinliang) capture build_from_cfg exception
        logger.error(e)
    else:
        # logger.info('Success in building learn rate scheduler')
        return lr_scheduler
    finally:
        # logger.info('build_lr_scheduler completion')
        pass


def modify_lr_config(lr_config, optimizer):
    from imix.utils.config import ConfigDict

    new_lr_config = ConfigDict()
    new_lr_config['type'] = 'WarmupMultiStepLR'
    new_lr_config['optimizer'] = optimizer
    new_lr_config['milestones'] = lr_config.step
    new_lr_config['warmup_factor'] = lr_config.warmup_ratio
    new_lr_config['warmup_iters'] = lr_config.warmup_iters
    new_lr_config['warmup_method'] = lr_config.warmup

    return new_lr_config
