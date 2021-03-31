# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os.path as osp
import pickle
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from torch.nn import Module

# import imix.utils.comm as comm
import imix.utils_mix.distributed_info as comm
# from imix.utils.checkpoint import Checkpointer
from imix.utils.file_io import PathManager
from imix.utils_mix.checkpoint import Checkpointer

# from .c2_model_loading import align_and_update_state_dicts #TODO(jinliang)

# class MixCheckpointer(Checkpointer):
#     """Same as :class:`Checkpointer`, but is able to handle models in
#     detectron.
#
#     & detectron2 model zoo, and apply conversions for legacy models.
#     """
#
#     def __init__(self,
#                  model,
#                  save_dir='',
#                  *,
#                  save_to_disk=None,
#                  **checkpointables):
#         is_main_process = comm.is_main_process()
#         super().__init__(
#             model,
#             save_dir,
#             save_to_disk=is_main_process
#             if save_to_disk is None else save_to_disk,
#             **checkpointables,
#         )
#
#     def _load_file(self, filename):
#         if filename.endswith('.pkl'):
#             with PathManager.open(filename, 'rb') as f:
#                 data = pickle.load(f, encoding='latin1')
#             if 'model' in data and '__author__' in data:
#                 # file is in Detectron2 model zoo format
#                 self.logger.info("Reading a file from '{}'".format(
#                     data['__author__']))
#                 return data
#             else:
#                 # assume file is from Caffe2 / Detectron1 model zoo
#                 if 'blobs' in data:
#                     # Detection models have "blobs", but ImageNet models don't
#                     data = data['blobs']
#                 data = {
#                     k: v
#                     for k, v in data.items() if not k.endswith('_momentum')
#                 }
#                 return {
#                     'model': data,
#                     '__author__': 'Caffe2',
#                     'matching_heuristics': True
#                 }
#
#         loaded = super()._load_file(filename)  # load native pth checkpoint
#         if 'model' not in loaded:
#             loaded = {'model': loaded}
#         return loaded
#
#     def _load_model(self, checkpoint):
#         if checkpoint.get('matching_heuristics', False):
#             self._convert_ndarray_to_tensor(checkpoint['model'])
#             # convert weights by name-matching heuristics
#             model_state_dict = self.model.state_dict()
#             # align_and_update_state_dicts(
#             #     model_state_dict,
#             #     checkpoint["model"],
#             #     c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
#             # )
#             checkpoint['model'] = model_state_dict
#         # for non-caffe2 models, use standard ways to load it
#         incompatible = super()._load_model(checkpoint)
#         if incompatible is None:  # support older versions of fvcore
#             return None
#
#         model_buffers = dict(self.model.named_buffers(recurse=False))
#         for k in ['pixel_mean', 'pixel_std']:
#             # Ignore missing key message about pixel_mean/std.
#             # Though they may be missing in old checkpoints, they will be correctly
#             # initialized from config anyway.
#             if k in model_buffers:
#                 try:
#                     incompatible.missing_keys.remove(k)
#                 except ValueError:
#                     pass
#         return incompatible


class MixCheckpointer(Checkpointer):

    def __int__(self,
                model: Module,
                save_dir: str = '',
                *,
                is_save_disk=None,
                is_record_ck: bool = True,
                **other_train_info: object):
        is_master_process = comm.is_main_process()
        if is_save_disk is None:
            is_save_disk = is_master_process
        super().__init__(
            model=model, save_dir=save_dir, is_save_disk=is_save_disk, is_record_ck=is_record_ck, **other_train_info)

    def _load_file_from_path(file_path: str):
        """
        Support multiple formats to load,such as pkl,currently only supports pth
        :return:
        """
        file_format = osp.splitext(file_path)[-1]
        if file_format != '.pth':
            raise Exception('the format of file_path:{} is {},currently only supports pth'.format(
                file_path, file_format))
        checkpoint = super()._load_file_from_path(file_path=file_path)  # load native checkpoint
        if 'model' not in checkpoint:
            checkpoint = {'model': checkpoint}
        return checkpoint

    def _load_model(self, checkpoint: Any):
        """Enhance the  compatibility of the loaded model by ignoring the
        missing key message in checkpoint.

        :param checkpoint:
        :return:
        """
        if checkpoint.get('matching_heuristics', False):
            self._state_dict_to_tensor(checkpoint['model'])
            checkpoint['model'] = self.model.state_dict()

        incompatible_keys = super()._load_model(checkpoint)
        if incompatible_keys is None:
            return None
        else:
            return incompatible_keys
