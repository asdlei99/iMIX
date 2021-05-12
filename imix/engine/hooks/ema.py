from .base_hook import HookBase
from .builder import HOOKS
import imix.utils_imix.distributed_info as comm


@HOOKS.register_module()
class EMAIterHook(HookBase):

    def __init__(self, config):
        super().__init__()
        self.use_ema = config.use_ema
        self.ema_decay_ratio = config.ema_decay_ratio

    def before_train(self):
        if self.use_ema:
            self.ema_state_dict = {}
            for param_name, param_tensor in self.trainer.model.state_dict().items():
                # we currently store the ema params on GPU
                self.ema_state_dict[param_name] = param_tensor.clone().detach()

    def after_train_iter(self):
        if self.use_ema:
            self.trainer.model.zero_grad()
            for param_name, param_tensor in self.trainer.model.state_dict().items():
                assert param_name in self.ema_state_dict
                self.ema_state_dict[param_name] -= (1.0 - self.ema_decay_ratio) * (
                    self.ema_state_dict[param_name] - param_tensor)


@HOOKS.register_module()
class EMAEpochHook(HookBase):

    def __init__(self, config):
        self.use_ema = config.use_ema
        self.ema_decay_ratio = config.ema_decay_ratio
        self.bkp_state_dict = None
        self.savePath = 'tttt'  # todo from trainer
        # self.ema_state_dict = # todo from EMAIterHook

    def before_train_epoch(self):
        # If EMA is used, we use averaged model to make eval
        if self.use_ema:
            # backup current params on cpu
            self.bkp_state_dict = {
                param_name: param_tensor.cpu().detach()
                for param_name, param_tensor in self.trainer.model.state_dict().items()
            }
            # load averaged params
            self.trainer.model.load_state_dict(self.ema_state_dict)

    def after_train_epoch(self):
        # If EMA is used, recover unaveraged params
        if self.use_ema:
            self.trainer.model.load_state_dict(self.bkp_state_dict)

            if comm.is_main_process():
                output_ema_state_dict = {}
                for param_name in self.trainer.model.state_dict():
                    assert param_name in self.ema_state_dict
                    if hasattr(model, 'module'):
                        output_ema_state_dict[param_name[7:]] = self.ema_state_dict[param_name]  # skip prefix "module."
                    else:
                        output_ema_state_dict[param_name] = self.ema_state_dict[param_name]
                output_ema_model_file = os.path.join(self.savePath, 'pytorch_model_' + str(epochId) + '_ema.bin')
                torch.save(output_ema_state_dict, output_ema_model_file)
