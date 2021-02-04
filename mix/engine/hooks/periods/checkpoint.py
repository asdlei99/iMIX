from ..base_hook import HookBase
# from mix.utils.checkpoint import PeriodicCheckpointer
from mix.utils_mix.checkpoint import PeriodicCheckpointer
from ..builder import HOOKS
# import mix.utils.comm as comm
import mix.utils_mix.distributed_info as comm


@HOOKS.register_module()
class CheckPointHook(PeriodicCheckpointer, HookBase):

  def before_train(self):
    self.max_iter = self.trainer.max_iter
    if self.trainer.by_epoch:
      self.max_epoch = self.trainer.max_epoch

  def after_train_iter(self):  # TODO(jinliang):modify
    if not self.trainer.by_epoch:
      # self.step(self.trainer.iter) # step(old) --> record_iter_checkpoint
      iter_other_info = {'by_epoch', self.trainer.by_epoch}
      self.record_iter_checkpoint(self.trainer.iter, **iter_other_info)

  def after_train_epoch(self):
    # self.save(name='epoch_{}'.format(self.trainer.epoch)) # save(old) --> record_epoch_checkpoint
    epoch_other_info = {
        'epoch_inner_iter': self.trainer.inner_iter,
        'epoch_iter': self.trainer.iter,
        'by_epoch': self.trainer.by_epoch
    }
    self.record_epoch_checkpoint(self.trainer.epoch, **epoch_other_info)

  def _multi_gpus_sync(self):
    comm.synchronize()
