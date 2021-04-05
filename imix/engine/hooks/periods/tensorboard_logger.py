# TODO(jinliang):jinliang_copy
# from .log_buffer import LogBufferWriter, get_log_buffer
from .log_buffer_imix import get_log_buffer, LogBufferWriter
from ..builder import HOOKS

# @HOOKS.register_module()
# class TensorboardXLoggerHook(LogBufferWriter):
#
#     def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
#         self._tb_writer = SummaryWriter(log_dir, kwargs)
#         self._window_size = window_size
#
#     def write(self):
#         log_buffer = get_log_buffer()
#         for k, v in log_buffer.get_latest_smoothing_hint(
#                 self._window_size).items():
#             self._tb_writer.add_scalar(k, v, log_buffer._iter)
#         #TODO(jinliang) add_image
#         #TODO(jinliang) add_histogram_raw
#
#     def close(self):
#         self._tb_writer.close()

# @HOOKS.register_module()
# class TensorboardXLoggerHook(LogBufferWriter):
#     """Write all scalars to a tensorboard file."""
#
#     def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
#         """
#         Args:
#             log_dir (str): the directory to save the output events
#             window_size (int): the scalars will be median-smoothed by this window size
#
#             kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
#         """
#         self._window_size = window_size
#         from torch.utils.tensorboard import SummaryWriter
#
#         self._writer = SummaryWriter(log_dir, **kwargs)
#
#     def write(self):
#         storage = get_log_buffer()
#         for k, v in storage.latest_with_smoothing_hint(
#                 self._window_size).items():
#             self._writer.add_scalar(k, v, storage.iter)
#
#         # storage.put_{image,histogram} is only meant to be used by
#         # tensorboard writer. So we access its internal fields directly from here.
#         if len(storage._vis_data) >= 1:
#             for img_name, img, step_num in storage._vis_data:
#                 self._writer.add_image(img_name, img, step_num)
#             # Storage stores all image data and rely on this writer to clear them.
#             # As a result it assumes only one writer will use its image data.
#             # An alternative design is to let storage store limited recent
#             # data (e.g. only the most recent image) that all writers can access.
#             # In that case a writer may not see all image data if its period is long.
#             storage.clear_images()
#
#         if len(storage._histograms) >= 1:
#             for params in storage._histograms:
#                 self._writer.add_histogram_raw(**params)
#             storage.clear_histograms()
#
#     def close(self):
#         if hasattr(self,
#                    '_writer'):  # doesn't exist when the code fails at import
#             self._writer.close()


@HOOKS.register_module()
class TensorboardLoggerHook(LogBufferWriter):
    """Write all scalars to a tensorboard file."""

    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size

            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir + '/runs/', **kwargs)

    # def write(self):
    #     storage = get_log_buffer()
    #     for k, v in storage.latest_with_smoothing_hint(
    #             self._window_size).items():
    #         self._writer.add_scalar(k, v, storage.iter)
    #
    #     # storage.put_{image,histogram} is only meant to be used by
    #     # tensorboard writer. So we access its internal fields directly from here.
    #     if len(storage._vis_data) >= 1:
    #         for img_name, img, step_num in storage._vis_data:
    #             self._writer.add_image(img_name, img, step_num)
    #         # Storage stores all image data and rely on this writer to clear them.
    #         # As a result it assumes only one writer will use its image data.
    #         # An alternative design is to let storage store limited recent
    #         # data (e.g. only the most recent image) that all writers can access.
    #         # In that case a writer may not see all image data if its period is long.
    #         storage.clear_images()
    #
    #     if len(storage._histograms) >= 1:
    #         for params in storage._histograms:
    #             self._writer.add_histogram_raw(**params)
    #         storage.clear_histograms()

    def process_buffer_data(self):
        self._add_scalar()
        self._add_image()
        self._add_histogram()

    def _add_scalar(self):
        for name, value in self.log_buffer.latest_with_smoothing_hint(self._window_size).items():
            self._writer.add_scalar(name, value, self.log_buffer.iter)

    def _add_image(self):
        if len(self.log_buffer.vis_data) >= 1:
            for img_name, img_data, step_num in self.log_buffer.vis_data.images:
                self._writer.add_image(img_name, img_data, step_num)
            self.log_buffer.clear_images()

    def _add_histogram(self):
        if len(self.log_buffer.histograms) >= 1:
            for hist_params in self.log_buffer.histograms.histograms:
                self._writer.add_histogram_raw(**hist_params)
            self.log_buffer.clear_histograms()

    def close(self):
        if hasattr(self, '_writer'):  # doesn't exist when the code fails at import
            self._writer.close()
