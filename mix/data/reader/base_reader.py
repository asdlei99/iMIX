"""
author: lxc
created time: 2020/8/17
"""

import numpy as np
import lmdb
import os
import pickle


class BaseDataReader():

  def __init__(self, cfg):
    # load config: path, name, split ...
    pass

  def load(self):
    pass

  def __getitem__(self, item):
    pass

  def deduplist(self, l):
    return list(set(l))


class MMFDataReader(BaseDataReader):

  def __init__(self, cfg):
    self.init_default_params(cfg)

    self.has_mix_global = cfg.get('mix_global_features') is not None
    self.has_mix_ocr = cfg.get('mix_ocr_features') is not None

    self.mix_features_dir = []
    self.mix_ocr_features_dir = []
    self.mix_global_features_dir = []
    self.mix_annotations_path = []
    self.mix_annotations = []
    self.item_splits = []
    for data in self.splits:
      self.mix_features_dir.append(cfg.mix_features[data])
      self.mix_annotations_path.append(cfg.mix_annotations[data])
      annotations_single_split = np.load(
          cfg.mix_annotations[data], allow_pickle=True)[1:]
      self.mix_annotations.extend(annotations_single_split)
      self.item_splits.extend([data] * len(annotations_single_split))
      if self.has_mix_global:
        self.mix_global_features_dir.append(cfg.mix_global_features[data])
      if self.has_mix_ocr:
        self.mix_ocr_features_dir.append(cfg.mix_ocr_features[data])

    if self.default_feature:
      self.feature_txns = []
      self.feature_global_txns = []
      self.feature_ocr_txns = []
      for mix_feature_dir in set(self.mix_features_dir):
        self.feature_txns.append(lmdb.open(mix_feature_dir).begin())
      for mix_feature_global_dir in set(self.mix_global_features_dir):
        self.feature_global_txns.append(
            lmdb.open(mix_feature_global_dir).begin())
      for mix_feature_ocr_dir in set(self.mix_ocr_features_dir):
        self.feature_ocr_txns.append(lmdb.open(mix_feature_ocr_dir).begin())
    else:
      self.features_pathes = {}
      for split in self.splits:
        mix_features_dir_tmp = self.mix_features_dir[self.splits.index(split)]
        names = os.listdir(mix_features_dir_tmp)
        for name in names:
          self.features_pathes[split + '_' +
                               name.split('.pth')[0]] = os.path.join(
                                   mix_features_dir_tmp, name)

  def get_featureinfo_from_txns(self, txns, key):
    feature_info = None
    key = str(key)
    for txn in txns:
      feature_info = txn.get(key.encode())
      if feature_info is not None:
        break
    return None if feature_info is None else pickle.loads(feature_info)

  def init_default_params(self, cfg):
    self.card = cfg.get('card', 'default')
    assert self.card in ['default', 'grid']
    self.default_feature = self.card == 'default'
    splits = cfg.datasets
    if isinstance(splits, str):
      splits = [splits]
    self.splits = splits
