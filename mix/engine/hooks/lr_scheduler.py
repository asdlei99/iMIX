from collections import Counter

from .base_hook import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class LRSchedulerHook(HookBase):
    """A hook which executes a torch builtin LR scheduler and summarizes the
    LR.

    It is executed after every iteration.
    """

    def __init__(self, optimizer, scheduler):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g['params']) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g['lr'] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g['lr'] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g['params']) == largest_group:
                    self._best_param_group_id = i
                    break

    def after_iter(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]['lr']
        self.trainer.log_buffer.put_scalar('lr', lr, smoothing_hint=False)
        self._scheduler.step()
