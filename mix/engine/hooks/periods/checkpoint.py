from ..base_hook import HookBase
from mix.utils.checkpoint import PeriodicCheckpointer
from ..builder import HOOKS
# import mix.utils.comm as comm
import mix.utils_mix.distributed_info as comm


@HOOKS.register_module()
class CheckPointHook(PeriodicCheckpointer, HookBase):

    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_train_iter(self):  # TODO(jinliang):modify
        if self.trainer.by_epoch is False:
            self.step(self.trainer.iter)

    def after_train_epoch(self):
        self.save(name='epoch_{}'.format(self.trainer.epoch))

    def _multi_gpus_sync(self):
        comm.synchronize()
