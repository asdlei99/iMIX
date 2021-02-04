# TODO(jinliang):jinliang_copy
import logging
from mix.utils.precise_bn import get_bn_modules, update_bn_stats
from .base_hook import HookBase
import itertools
from .periods.log_buffer import LogBufferStorage
from .builder import HOOKS


@HOOKS.register_module()
class PreciseBNHook(HookBase):
  """The standard implementation of BatchNorm uses EMA in inference, which is
  sometimes suboptimal. This class computes the true average of statistics
  rather than the moving average, and put true averages to every BN layer in
  the given model.

  It is executed every ``period`` iterations and after the last iteration.
  """

  def __init__(self, period, model, data_loader, num_iter):
    """
        Args:
            period (int): the period this hook is run, or 0 to not run during training.
                The hook will always run in the end of training.
            model (nn.Module): a module whose all BN layers in training mode will be
                updated by precise BN.
                Note that user is responsible for ensuring the BN layers to be
                updated are in training mode when this hook is triggered.
            data_loader (iterable): it will produce data to be run by `model(data)`.
            num_iter (int): number of iterations used to compute the precise
                statistics.
        """
    self._logger = logging.getLogger(__name__)
    if len(get_bn_modules(model)) == 0:
      self._logger.info(
          'PreciseBN is disabled because model does not contain BN layers in training mode.'
      )
      self._disabled = True
      return

    self._model = model
    self._data_loader = data_loader
    self._num_iter = num_iter
    self._period = period
    self._disabled = False

    self._data_iter = None

  def after_train_iter(self):
    next_iter = self.trainer.iter + 1
    is_final = next_iter == self.trainer.max_iter
    if is_final or (self._period > 0 and next_iter % self._period == 0):
      self.update_stats()

  def update_stats(self):
    """Update the model with precise statistics.

    Users can manually call this method.
    """
    if self._disabled:
      return

    if self._data_iter is None:
      self._data_iter = iter(self._data_loader)

    def data_loader():
      for num_iter in itertools.count(1):
        if num_iter % 100 == 0:
          self._logger.info('Running precise-BN ... {}/{} iterations.'.format(
              num_iter, self._num_iter))
        # This way we can reuse the same iterator
        yield next(self._data_iter)

    with LogBufferStorage():  # capture events in a new storage to discard them
      self._logger.info(
          'Running precise-BN for {} iterations...  '.format(self._num_iter) +
          'Note that this could produce different statistics every time.')
      update_bn_stats(self._model, data_loader(), self._num_iter)
