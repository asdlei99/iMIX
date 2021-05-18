import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from imix.utils_imix.config import seed_all_rng, imixEasyDict
from imix.utils_imix.registry import Registry, build_from_cfg
from .sampler import build_sampler, build_batch_sampler
from typing import Optional

VOCAB = Registry('vocab')
PREPROCESSOR = Registry('preprocessor')
DATASETS = Registry('dataset')

PROCESSOR = Registry('processor')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_vocab(cfg):
    """Build vocab."""
    return build_from_cfg(cfg, VOCAB)


def build_preprocessor(cfg):
    """Build preprocessor."""
    return build_from_cfg(cfg, PREPROCESSOR)


def build_dataset(dataset_cfg, default_args=None):
    dataset = build_from_cfg(dataset_cfg, DATASETS, default_args)
    return dataset


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_data_loader_by_epoch(dataset, cfg, is_training=True):

    def get_cfg_param(data_cfg):
        batch_size = getattr(data_cfg, 'samples_per_gpu')
        num_workers = getattr(data_cfg, 'workers_per_gpu')
        drop_last = getattr(data_cfg, 'drop_last', False)
        pin_memory = getattr(data_cfg, 'pin_memory', False)
        sampler_cfg = getattr(data_cfg, 'sampler', None)
        batch_sampler_cfg = getattr(data_cfg, 'batch_sampler', None)
        shuffle = getattr(data_cfg, 'shuffle', True if is_training else False)
        return batch_size, num_workers, drop_last, pin_memory, sampler_cfg, batch_sampler_cfg, shuffle

    batch_size, num_workers, drop_last, pin_memory, sampler_cfg, batch_sampler_cfg, shuffle = get_cfg_param(
        cfg.train_data if is_training else cfg.test_data)

    dataloader_param = {'dataset': dataset, 'pin_memory': pin_memory, 'num_workers': num_workers}
    if sampler_cfg:
        sampler_cfg = imixEasyDict({'type': sampler_cfg}) if isinstance(sampler_cfg, str) else sampler_cfg
        sampler = build_sampler(sampler_cfg, default_args={'dataset': dataset})
        dataloader_param.update({'batch_size': batch_size, 'sampler': sampler, 'drop_last': drop_last})
    elif batch_sampler_cfg:
        batch_sampler = build_batch_sampler(batch_sampler_cfg, default_args={'dataset': dataset})
        collate_fn = getattr(dataset, 'collate_fn', None)
        dataloader_param.update({'batch_sampler': batch_sampler, 'collate_fn': collate_fn})
    else:
        dataloader_param.update({'batch_size': batch_size, 'drop_last': drop_last})

    return DataLoader(**dataloader_param)


def batch_collate_fn(batch):
    return batch


def worker_init_reset_seed(worker_id):
    return seed_all_rng(np.random.randint(2**31) + worker_id)


def build_imix_train_loader(cfg):
    """A data loader is created  by the following steps:

    1. Use the dataset names in config to create dataset
    2. Build PyTorch DataLoader
    """

    dataset = build_dataset(cfg.train_data.data)
    return build_data_loader_by_epoch(dataset, cfg, is_training=True)


def build_imix_test_loader(cfg, dataset_name: Optional[str] = ''):  # TODO(jinliang)
    dataset = build_dataset(cfg.test_data.data)
    return build_data_loader_by_epoch(dataset, cfg, is_training=False)
