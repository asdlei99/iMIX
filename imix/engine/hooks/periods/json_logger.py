# TODO(jinliang):jinliang_copy
import json
import os

from imix.utils_imix.file_io import PathManager
from ..builder import HOOKS
from .log_buffer_imix import LogBufferWriter

# @HOOKS.register_module()
# class JSONLoggerHook(LogBufferWriter):
#
#     def __init__(self,
#                  json_file_path: str,
#                  window_size: int = 20,
#                  smoothing_method: str = ''):
#         self._json_file = json_file_path
#         self._window_size = window_size
#         self._smoothing_method = smoothing_method
#
#     def write(self):
#         pass
#
#     def close(self):
#         self._json_file.close()

# @HOOKS.register_module()
# class JSONLoggerHook(LogBufferWriter):
#     """Write scalars to a json file.
#
#     It saves scalars as one json per line (instead of a big json) for easy parsing.
#
#     Examples parsing such a json file:
#
#     .. code-block:: none
#
#         $ cat metrics.json | jq -s '.[0:2]'
#         [
#           {
#             "data_time": 0.008433341979980469,
#             "iteration": 20,
#             "loss": 1.9228371381759644,
#             "loss_box_reg": 0.050025828182697296,
#             "loss_classifier": 0.5316952466964722,
#             "loss_mask": 0.7236229181289673,
#             "loss_rpn_box": 0.0856662318110466,
#             "loss_rpn_cls": 0.48198649287223816,
#             "lr": 0.007173333333333333,
#             "time": 0.25401854515075684
#           },
#           {
#             "data_time": 0.007216215133666992,
#             "iteration": 40,
#             "loss": 1.282649278640747,
#             "loss_box_reg": 0.06222952902317047,
#             "loss_classifier": 0.30682939291000366,
#             "loss_mask": 0.6970193982124329,
#             "loss_rpn_box": 0.038663312792778015,
#             "loss_rpn_cls": 0.1471673548221588,
#             "lr": 0.007706666666666667,
#             "time": 0.2490077018737793
#           }
#         ]
#
#         $ cat metrics.json | jq '.loss_mask'
#         0.7126231789588928
#         0.689423680305481
#         0.6776131987571716
#         ...
#     """
#
#     def __init__(self, json_file, window_size=20):
#         """
#         Args:
#             json_file (str): path to the json file. New data will be appended if the file exists.
#             window_size (int): the window size of median smoothing for the scalars whose
#                 `smoothing_hint` are True.
#         """
#         self._file_handle = PathManager.open(json_file, 'a')
#         self._window_size = window_size
#
#     def write(self):  # TODO(jinliang):modify
#         storage = get_log_buffer()
#         to_save = {'iteration': storage.iter}
#         to_save.update(storage.latest_with_smoothing_hint(self._window_size))
#         self._file_handle.write(json.dumps(to_save, sort_keys=True) + '\n')
#         self._file_handle.flush()
#         try:
#             os.fsync(self._file_handle.fileno())
#         except AttributeError:
#             pass
#
#     def close(self):
#         self._file_handle.close()


@HOOKS.register_module()
class JSONLoggerHook(LogBufferWriter):
    """Write scalars to a json file.

    It saves scalars as one json per line (instead of a big json) for easy parsing.

    Examples parsing such a json file:

    .. code-block:: none

        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 20, or 'epoch':1
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 40,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]

        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...
    """

    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file. New data will be appended if the file exists.
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
        """
        self._file_handle = PathManager.open(json_file, 'w')
        self._window_size = window_size

    # def write(self):  # TODO(jinliang):modify
    #     data = self._update_data()
    #     self._file_handle.write(json.dumps(data, sort_keys=True) + '\n')
    #     self._file_handle.flush()
    #     try:
    #         os.fsync(self._file_handle.fileno())
    #     except AttributeError:
    #         pass

    def close(self):
        if hasattr(self, '_file_handle'):
            self._file_handle.close()

    # def _update_data(self):
    #     storage = get_log_buffer()
    #     if storage.by_epoch:
    #         data = {"epoch": storage.epoch, "inner-iteration": storage.iter}
    #     else:
    #         data = {'iteration': storage.iter}
    #     data.update(storage.latest_with_smoothing_hint(self._window_size))
    #     return data

    def process_buffer_data(self):

        if self.log_buffer.by_epoch:
            data = {'epoch': self.log_buffer.epoch, 'inner-iteration': self.log_buffer.iter}
        else:
            data = {'iteration': self.log_buffer.iter}

        data.update(self.log_buffer.latest_with_smoothing_hint(self._window_size))
        self._file_handle.write(json.dumps(data, sort_keys=True) + '\n')
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass
