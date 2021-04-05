from .base_hook import HookBase
from .builder import HOOKS
from torch.autograd import set_detect_anomaly


@HOOKS.register_module()
class AutogradAnomalyDetectHook(HookBase):
    """A hook that anomaly detection for the autograd engine`.

    NOTE:
        This hook should be enabled only for debugging as the different tests will slow down your program execution.
    """

    def __init__(self, mode: bool = False):
        self._mode = mode

    def before_train(self):
        set_detect_anomaly(mode=self._mode)
