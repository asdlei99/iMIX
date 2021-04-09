# from imix.utils.registry import Registry, build_from_cfg
import logging
import random
from functools import partial

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler

# import imix.utils.comm as comm
import imix.utils_imix.distributed_info as comm
# from .sampler import DistributedGroupSampler, DistributedSampler, GroupSampler
from imix.utils.dist_utils import get_dist_info
# from imix.utils.comm import get_world_size
# import imix.utils_imix.distributed_info as comm
from imix.utils.env import seed_all_rng
from imix.utils_imix.registry import Registry, build_from_cfg
from .parallel.collate import collate
from .sampler import InferenceSampler, TrainingSampler

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


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=False,
                     seed=None,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    from .sampler.group_sampler import DistributedGroupSampler, GroupSampler
    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = DistributedGroupSampler(dataset, samples_per_gpu, world_size, rank)
        else:
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(
            dataset, samples_per_gpu) if shuffle else None  # TODO(jinliang) mmdet:flag according to image aspect ratio
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# def build_mix_train_loader(cfg):
#     """
#     A data loader is created  by the following steps:
#     1. Use the dataset names in config to create dataset
#     2. Build PyTorch DataLoader
#     """
#
#     dataset = build_dataset(cfg.train_data.data)
#     data_loader = build_dataloader(
#         dataset,
#         cfg.train_data.samples_per_gpu,
#         cfg.train_data.workers_per_gpu,
#         cfg.num_gpus,
#         # dist=cfg.distributed,
#         seed=cfg.SEED)
#     return data_loader


def build_data_loader_by_iter(dataset, cfg, is_training=True):
    logger = logging.getLogger(__name__)
    if is_training:
        sampler_name = cfg.train_data.sampler_name
        logger.info('using training sampler {}'.format(sampler_name))
    else:
        sampler_name = cfg.test_data.sampler_name
        logging.info('using testing sampler {}'.format(sampler_name))

    if sampler_name == 'TrainingSampler':
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == 'TestingSampler':
        sampler = InferenceSampler(len(dataset))
    else:
        raise ValueError('Unimplemented  sampling method: {}'.format(sampler_name))

    # batch_collate_fn = lambda batch: batch
    # worker_init_reset_seed = lambda worker_id: seed_all_rng(np.random.randint(2 ** 31) + worker_id)

    if is_training:
        # world_size = get_world_size()
        # total_batch_size = world_size * cfg.train_data.samples_per_gpu  # multimachine ? *
        # assert (
        #         total_batch_size > 0 and total_batch_size % world_size == 0
        # ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        #     total_batch_size, world_size
        # )
        # batch_size = total_batch_size // world_size

        batch_size = cfg.train_data.samples_per_gpu
        num_workers = cfg.train_data.workers_per_gpu
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
        data_loader = DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=batch_collate_fn,
            worker_init_fn=worker_init_reset_seed)
    else:
        num_workers = cfg.test_data.workers_per_gpu
        batch_sampler = BatchSampler(sampler, cfg.test_data.samples_per_gpu, drop_last=False)
        data_loader = DataLoader(
            dataset, num_workers=num_workers, batch_sampler=batch_sampler, collate_fn=batch_collate_fn)

    return data_loader


def build_data_loader_by_epoch(dataset, cfg, is_training=True):
    batch_size = cfg.train_data.samples_per_gpu if is_training else cfg.test_data.samples_per_gpu
    num_workers = cfg.train_data.workers_per_gpu if is_training else cfg.test_data.workers_per_gpu
    drop_last = cfg.train_data.get('drop_last', False) if is_training else cfg.test_data.get('drop_last', False)
    shuffle = cfg.train_data.get('shuffle', False) if is_training else cfg.test_data.get('shuffle', False)

    if comm.get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=True)
        return DataLoader(
            dataset=dataset,
            pin_memory=False,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            shuffle=shuffle)
    else:
        return DataLoader(
            dataset=dataset,
            pin_memory=False,
            num_workers=num_workers,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle)


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
    if hasattr(cfg, 'by_iter'):
        return build_data_loader_by_iter(dataset, cfg, is_training=True)
    else:
        return build_data_loader_by_epoch(dataset, cfg, is_training=True)


# def build_mix_test_loader(cfg, dataset_name):  # TODO(jinliang)
#     dataset = build_dataset(cfg.test_data.data)
#     data_loader = build_dataloader(dataset, cfg.test_data.samples_per_gpu,
#                                    cfg.test_data.workers_per_gpu,
#                                    cfg.num_gpus)
#     # dist=cfg.distributed)data
#     return data_loader


def build_imix_test_loader(cfg, dataset_name):  # TODO(jinliang)
    dataset = build_dataset(cfg.test_data.data)
    if hasattr(cfg, 'by_iter'):
        return build_data_loader_by_iter(dataset, cfg, is_training=False)
    else:
        return build_data_loader_by_epoch(dataset, cfg, is_training=False)
