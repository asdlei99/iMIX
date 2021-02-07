from torch.utils.data import Dataset, IterableDataset
import yaml
from ..reader.vqa_reader import VQAReader
from ..infocomp.vqa_infocpler import VQAInfoCpler
from ..builder import DATASETS
import numpy as np
import torch

import logging
# import imix.utils.comm as comm
import imix.utils_imix.distributed_info as comm

# VQA_PATH_CONFIG = yaml.load(open("datasets/dataset_vqa.yaml"))["dataset_configs"]

# @DATASETS.register_module()
# class VQADATASET(Dataset):
#     def __init__(self, splits):
#         self.reader = VQAReader(VQA_PATH_CONFIG, splits)
#         self.infocpler = VQAInfoCpler(VQA_PATH_CONFIG)
#
#     def __len__(self):
#         return len(self.reader)
#
#     def __getitem__(self, item):
#
#         itemFeature = self.reader[item]
#         itemFeature = self.infocpler.completeInfo(itemFeature)
#         return itemFeature.feature, itemFeature.input_ids, itemFeature.answers_scores, itemFeature.input_mask


def remove_None_value_elements(input_dict):
  if type(input_dict) is not dict:
    return None
  result = {}
  for key in input_dict:
    tmp = {}
    if input_dict[key] is not None:
      if type(input_dict[key]).__name__ == 'dict':
        tmp.update({key: remove_None_value_elements(input_dict[key])})
      else:
        tmp.update({key: input_dict[key]})
    result.update(tmp)
  return result


@DATASETS.register_module()
class VQADATASET(Dataset):

  def __init__(self, reader, info_cpler, limit_nums=None):
    if comm.is_main_process():
      logger = logging.getLogger(__name__)
      logger.info('start loading vqadata')

    self.reader = VQAReader(reader)
    self.infocpler = VQAInfoCpler(info_cpler)
    self._limit_sample_nums = limit_nums
    self.splits = reader.datasets
    if comm.is_main_process():
      logger.info('load vqadata {} successfully'.format(reader.datasets))

  def __len__(self):
    if self._limit_sample_nums and self._limit_sample_nums > 0:
      return min(len(self.reader), self._limit_sample_nums)
    return len(self.reader)

  def __getitem__(self, idx):
    # idx = 0
    itemFeature = self.reader[idx]
    itemFeature = self.infocpler.completeInfo(itemFeature)

    # TODO(jinliang+ce@lxc)
    item = {
        'feature': itemFeature.features,  # feature - feature
        'cls_prob': itemFeature.cls_prob,  # 1601 cls_prob
        'bbox': itemFeature.bbox,  # feature - bbox
        'image_dim': itemFeature.num_boxes,  # feature - bbox_Num
        'input_ids': itemFeature.input_ids,  # tokens - ids
        'input_mask': itemFeature.input_mask,  # tokens - mask
        'input_segment': itemFeature.input_segment,  # tokens - segments
        'input_lm_label_ids':
            itemFeature.input_lm_label_ids,  # tokens - mlm labels
        'question_id': itemFeature.question_id,
        'image_id': itemFeature.image_id,
    }

    if itemFeature.answers_scores is not None:
      item['answers_scores'] = itemFeature.answers_scores
    # return itemFeature.feature, itemFeature.input_ids, itemFeature.answers_scores, itemFeature.input_mask

    if 'test' in self.splits or 'oneval' in self.splits:
      item['quesid2ans'] = self.infocpler.qa_id2ans

    return item

    # return itemFeature
