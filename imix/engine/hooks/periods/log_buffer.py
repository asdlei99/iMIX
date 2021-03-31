# TODO(jinliang):jinliang_copy
import weakref
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from contextlib import contextmanager

import torch

from imix.utils.history_buffer import HistoryBuffer

# TODO(jinliang) logBufferStorage在_CURRENT_LOG_BUFFER_STACK存入数据,
# 而LogBufferWriter将存入数据写文件或终端输出，因此所有logBuffer数据仅有一份，
# 后期可通过带有装饰器单例模式进行优化

_CURRENT_LOG_BUFFER_STACK = []


def get_log_buffer():
    assert len(
        _CURRENT_LOG_BUFFER_STACK), "get_log_buffer() has to be called inside a 'with LogBufferStorage(...)' context!"
    return _CURRENT_LOG_BUFFER_STACK[-1]


# before_modification
# class LogBufferWriter:
#     """将LogBufferStorage数据按照不同类型writer."""
#
#     @abstractmethod
#     def write(self):
#         raise NotImplementedError
#
#     @abstractmethod
#     def close(self):
#         pass


class LogBufferWriter:
    """将LogBufferStorage数据按照不同类型writer."""

    def write(self):
        self.get_buffer_data()
        self.process_buffer_data()

    def close(self):
        pass

    def get_buffer_data(self):
        self.log_buffer = get_log_buffer()

    def process_buffer_data(self):
        raise NotImplementedError


# class LogBufferStorage:
#
#     def __init__(self, start_iter=0):
#         self._iter = start_iter
#         self._histograms = []
#
#     def push_image(self, img_name, img_data):
#         pass
#
#     def push_scalar(self, scalar_name, scalar_value, smooth_flag=True):
#         pass
#
#     def push_scalars(self, *, smooth_flag=True, **kwargs):
#         for name, scalar in kwargs.items():
#             self.push_scalar(name, scalar, smooth_flag=smooth_flag)
#
#     def push_histogram(self,
#                        histogram_name,
#                        histogram_data,
#                        histogram_bins=1000):
#         pass
#
#     def get_history(self, name):
#         pass
#
#     def histories(self):
#         pass
#
#     def get_latest_scalars(self):
#         pass
#
#     def get_latest_smoothing_hint(self, window_size=20):
#         pass
#
#     def get_smooth_hints(self):
#         pass
#
#     def step(self):
#         pass
#
#     @property
#     def iter(self):
#         return self._iter
#
#     @property
#     def iteration(self):
#         #TODO(jinliang) 后向传播???
#         return self._iter
#
#     def __enter__(self):
#         _CURRENT_LOG_BUFFER_STACK.append(self)
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         assert _CURRENT_LOG_BUFFER_STACK[-1] == self
#         _CURRENT_LOG_BUFFER_STACK.pop()
#
#     @contextmanager
#     def name_scope(self, name):
#         pass
#
#     def clear_log_buffer_images(self):
#         pass
#
#     def clear_log_buffer_histograms(self):
#         self._histograms = []


class LogBufferStorage:
    """The user-facing class that provides metric storage functionalities.

    In the future we may add support for storing / logging other types of data if needed.
    """

    def __init__(self, start_iter=0, sigle_epoch_iters=0, by_epoch=True):
        """
        Args:
            start_iter (int): the iteration number to start with
        """
        self._history = defaultdict(HistoryBuffer)
        self._smoothing_hints = {}
        self._latest_scalars = {}
        self._iter = start_iter
        self._current_prefix = ''
        self._vis_data = []
        self._histograms = []
        self._epoch_inner_iter = 0
        self._epoch = 0
        self._single_epoch_iters = sigle_epoch_iters
        self._by_epoch = by_epoch

    def put_image(self, img_name, img_tensor):
        """Add an `img_tensor` associated with `img_name`, to be shown on
        tensorboard.

        Args:
            img_name (str): The name of the image to put into tensorboard.
            img_tensor (torch.Tensor or numpy.array): An `uint8` or `float`
                Tensor of shape `[channel, height, width]` where `channel` is
                3. The image format should be RGB. The elements in img_tensor
                can either have values in [0, 1] (float32) or [0, 255] (uint8).
                The `img_tensor` will be visualized in tensorboard.
        """
        self._vis_data.append((img_name, img_tensor, self._iter))

    def put_scalar(self, name, value, smoothing_hint=True):
        """Add a scalar `value` to the `HistoryBuffer` associated with `name`.

        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.

                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
        """
        name = self._current_prefix + name
        history = self._history[name]
        value = float(value)
        history.update(value, self._iter)
        self._latest_scalars[name] = value

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            assert (existing_hint == smoothing_hint), 'Scalar {} was put with a different smoothing_hint!'.format(name)
        else:
            self._smoothing_hints[name] = smoothing_hint

    def put_scalars(self, *, smoothing_hint=True, **kwargs):
        """Put multiple scalars from keyword arguments.

        Examples:

            storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
        for k, v in kwargs.items():
            self.put_scalar(k, v, smoothing_hint=smoothing_hint)

    def put_histogram(self, hist_name, hist_tensor, bins=1000):
        """Create a histogram from a tensor.

        Args:
            hist_name (str): The name of the histogram to put into tensorboard.
            hist_tensor (torch.Tensor): A Tensor of arbitrary shape to be converted
                into a histogram.
            bins (int): Number of histogram bins.
        """
        ht_min, ht_max = hist_tensor.min().item(), hist_tensor.max().item()

        # Create a histogram with PyTorch
        hist_counts = torch.histc(hist_tensor, bins=bins)
        hist_edges = torch.linspace(start=ht_min, end=ht_max, steps=bins + 1, dtype=torch.float32)

        # Parameter for the add_histogram_raw function of SummaryWriter
        hist_params = dict(
            tag=hist_name,
            min=ht_min,
            max=ht_max,
            num=len(hist_tensor),
            sum=float(hist_tensor.sum()),
            sum_squares=float(torch.sum(hist_tensor**2)),
            bucket_limits=hist_edges[1:].tolist(),
            bucket_counts=hist_counts.tolist(),
            global_step=self._iter,
        )
        self._histograms.append(hist_params)

    def history(self, name):
        """
        Returns:
            HistoryBuffer: the scalar history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError('No history metric available for {}!'.format(name))
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[name -> number]: the scalars that's added in the current iteration.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        """Similar to :meth:`latest`, but the returned values are either the
        un- smoothed original latest value, or a median of the given
        window_size, depend on whether the smoothing_hint is True.

        This provides a default behavior that other writers can use.
        """
        result = {}
        for k, v in self._latest_scalars.items():
            result[k] = self._history[k].median(window_size) if self._smoothing_hints[k] else v
        return result

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """User should call this function at the beginning of each iteration,
        to notify the storage of the start of a new iteration.

        The storage will then be able to associate the new data with the correct iteration number.
        """
        self._iter += 1
        self._latest_scalars = {}

    def epoch_step(self):
        self._epoch += 1

    def epoch_iter(self, inner_iter):
        self._epoch_inner_iter = inner_iter + 1

    @property
    def iter(self):
        return self._iter

    @property
    def epoch(self):
        return self._epoch

    @property
    def epoch_inner_iter(self):
        return self._epoch_inner_iter

    @property
    def single_epoch_iters(self):
        return self._single_epoch_iters

    @property
    def iteration(self):
        # for backward compatibility
        return self._iter

    @property
    def by_epoch(self):
        return self._by_epoch

    def __enter__(self):
        _CURRENT_LOG_BUFFER_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_LOG_BUFFER_STACK[-1] == self
        _CURRENT_LOG_BUFFER_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip('/') + '/'
        yield
        self._current_prefix = old_prefix

    def clear_images(self):
        """Delete all the stored images for visualization.

        This should be called after images are written to tensorboard.
        """
        self._vis_data = []

    def clear_histograms(self):
        """Delete all the stored histograms for visualization.

        This should be called after histograms are written to tensorboard.
        """
        self._histograms = []
