# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# TODO(jinliang):jinliang_copy
import math
from bisect import bisect, bisect_right
from typing import List

import torch
from torch.optim.lr_scheduler import LambdaLR

from .builder import LR_SCHEDULERS

# NOTE: PyTorch's LR scheduler interface uses names that assume the LR changes
# only on epoch boundaries. We typically use iteration based schedules instead.
# As a result, "epoch" (e.g., as in self.last_epoch) should be understood to mean
# "iteration" instead.

# FIXME: ideally this would be achieved with a CombinedLRScheduler, separating
# MultiStepLR with WarmupLR but the current LRScheduler design doesn't allow it.


@LR_SCHEDULERS.register_module()
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = 'linear',
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of' ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method, self.last_epoch, self.warmup_iters,
                                                   self.warmup_factor)
        return [
            base_lr * warmup_factor * self.gamma**bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


@LR_SCHEDULERS.register_module()
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = 'linear',
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method, self.last_epoch, self.warmup_iters,
                                                   self.warmup_factor)
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            base_lr * warmup_factor * 0.5 * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


@LR_SCHEDULERS.register_module()
class PythiaScheduler(LambdaLR):

    def __init__(self, optimizer, *args, **kwargs):
        self._lambda_func = lr_lambda_update

        super().__init__(optimizer, self.lr_lambda, *args, **kwargs)

    def lr_lambda(self, step):
        return self._lambda_func(step, self._global_config)


@LR_SCHEDULERS.register_module()
class MultiStepScheduler(PythiaScheduler):  #TODO(jinliang): modify

    def __init__(self, optimizer, *args, **kwargs):
        self.use_warmup = kwargs['use_warmup']
        self.lr_steps = kwargs['lr_steps']
        self.lr_ratio = kwargs['lr_ratio']
        self.warmup_iterations = kwargs['warmup_iterations'] if self.use_warmup else 0
        self.warmup_factor = kwargs['warmup_factor']
        assert self.warmup_iterations < self.lr_steps[0]
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iterations and self.use_warmup is True:
            alpha = float(self.last_epoch) / float(self.warmup_iterations)
            lr_ratio = self.warmup_factor * (1.0 - alpha) + alpha

            return [base_lr * lr_ratio for base_lr in self.base_lrs]
        else:
            return [base_lr * self.lr_ratio**bisect_right(self.lr_steps, self.last_epoch) for base_lr in self.base_lrs]


def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int, warmup_factor: float) -> float:
    """Return the learning rate warmup factor at a specific iteration. See
    :paper:`in1k1h` for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == 'constant':
        return warmup_factor
    elif method == 'linear':
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError('Unknown warmup method: {}'.format(method))


def lr_lambda_update(i_iter, cfg):
    if cfg.training.use_warmup is True and i_iter <= cfg.training.warmup_iterations:
        alpha = float(i_iter) / float(cfg.training.warmup_iterations)
        return cfg.training.warmup_factor * (1.0 - alpha) + alpha
    else:
        idx = bisect(cfg.training.lr_steps, i_iter)
        return pow(cfg.training.lr_ratio, idx)
