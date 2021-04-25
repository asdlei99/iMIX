import time

from torch.cuda.amp.autocast_mode import autocast
from .base_engine import EngineBase
from .hooks.periods import write_metrics
from .organizer import Organizer, is_mixed_precision
from imix.utils_imix.Timer import batch_iter


class CommonEngine(EngineBase):

    def __init__(self, model, data_loader, optimizer, loss_fn, batch_processor=None):
        super(CommonEngine, self).__init__()

        model.train()

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_processor = batch_processor

        self.data_loader = data_loader
        self.imixed_precision = False

    def run_train_iter(self, batch_data=None, data_time=None):
        assert self.model.training, '[CommonEngine] model was changed to eval model!'

        if self.batch_processor is not None:
            self.output = self.batch_processor(batch_data)  # TODO(jinliang) 暂时这么处理，缺少相关参数
        else:
            with autocast(enabled=is_mixed_precision()):  # TODO(jinliang) autocast warp
                self.model_output = self.model(
                    batch_data,
                    cur_epoch=getattr(self, 'epoch', None),
                    cur_iter=self.iter,
                    inner_iter=getattr(self, 'inner_iter', None))
                self.output = self.loss_fn(self.model_output)

        metrics_dict = {'data_time': data_time}
        metrics_dict.update(self.output)
        write_metrics(metrics_dict)

    def run_train_epoch(self):
        assert self.model.training, '[CommonEngine] model was changed to eval model!'
        time.sleep(2)  # prevent possible deadlockduring epoch transition
        for i, batch_data, data_time in batch_iter(self.data_loader):
            self.inner_iter = i
            self.before_train_iter()
            self.run_train_iter(batch_data, data_time)
            self.after_train_iter()
            self.iter += 1


class imixEngine(CommonEngine):

    def __init__(self, cfg):
        self.organizer = Organizer(cfg)
        super(imixEngine, self).__init__(
            model=self.organizer.model,
            data_loader=self.organizer.train_data_loader,
            optimizer=self.organizer.optimizer,
            loss_fn=self.organizer.losses_fn)

        self.start_iter = self.organizer.start_iter
        self.max_iter = self.organizer.max_iter
        self.start_epoch = self.organizer.start_epoch
        self.max_epoch = self.organizer.max_epoch
        self.cfg = self.organizer.cfg
        self.by_epoch = self.organizer.by_epoch
        self.is_lr_accumulation = self.organizer.is_lr_accumulation
        self.gradient_accumulation_steps = self.organizer.gradient_accumulation_steps

        self.imixed_precision = self.organizer.imixed_precision if hasattr(self.organizer,
                                                                           'imixed_precision') else False
        self.work_dir = cfg.work_dir

        self.register_hooks(self.organizer.hooks)

    def run_by_iter(self):
        super(imixEngine, self).train_by_iter(self.start_iter, self.max_iter)

    def run_by_epoch(self):
        super(imixEngine, self).train_by_epoch(self.start_epoch, self.max_epoch)

    def train(self):
        if self.by_epoch:
            self.run_by_epoch()
        else:
            self.run_by_iter()
