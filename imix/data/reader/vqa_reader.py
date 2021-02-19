"""
author: lxc
created time: 2020/8/18
"""

import numpy as np
import os
import torch
import lmdb
import pickle
from .base_reader import MMFDataReader
from ..utils.stream import ItemFeature
from ..utils.tokenization import BertTokenizer
from .base_reader import BaseDataReader


class VQAReader(MMFDataReader):

  def __init__(self, cfg):
    super().__init__(cfg)

  def __len__(self):
    return len(self.mix_annotations)

  def __getitem__(self, item):
    annotation = self.mix_annotations[item]
    split = self.item_splits[item]
    itemFeature = ItemFeature()
    itemFeature.error = False
    for k, v in annotation.items():
      itemFeature[k] = v

    # TODO(jinliang)
    # itemFeature.tokens = annotation["question_tokens"]
    # itemFeature.answers = annotation["answers"]
    # itemFeature.all_answers = annotation["all_answers"]
    # print(item)
    # itemFeature.ocr_tokens = annotation["ocr_tokens"]

    if split is not 'test':
      itemFeature.answers = annotation['answers']
      itemFeature.all_answers = annotation['all_answers']


    itemFeature.tokens = annotation['question_tokens']
    itemFeature.img_id = annotation['image_id']
    if self.default_feature:
      feature_info = None
      for txn in self.feature_txns:
        feature_info = pickle.loads(txn.get(annotation['image_name'].encode()))
        if feature_info is not None:
          break
      feature_global_info = None
      for txn in self.feature_global_txns:
        feature_global_info = pickle.loads(txn.get(annotation['image_name'].encode()))
        if feature_global_info is None:
          break
        else:
          feature_global_info['global_feature_path'] = feature_global_info.pop('feature_path')
          feature_global_info['global_features'] = feature_global_info.pop('features')
      if feature_info is None or feature_global_info is None:
        itemFeature.error = True
        itemFeature.feature = np.random.random((100, 2048))
        itemFeature.global_feature = np.random.random((100, 2048))
        return itemFeature

      for k, v in feature_info.items():
        itemFeature[k] = v
      for k, v in feature_global_info.items():
        itemFeature[k] = v
      return itemFeature
    feature_path = self.features_pathes[split + '_' + str(itemFeature.img_id)]
    itemFeature.feature = torch.load(feature_path)[0]
    return itemFeature
