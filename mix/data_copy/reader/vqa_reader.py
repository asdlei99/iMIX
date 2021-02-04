"""
author: lxc
created time: 2020/8/18
"""

import yaml
import numpy as np
import os
import torch
from collections import OrderedDict
import collections
from .base_reader import BaseDataReader
from ..vqadata.stream import ItemFeature

VQA_INFO_MAPPING = {
    'split_idx': 'split_idx',
    'image_id': 'img_id',
    'image_width': 'img_width',
    'image_height': 'img_height',
    'question': 'question',
    'answers': 'answers',
    'valid_answers': 'valid_answers'
}


class VQAReader(BaseDataReader):
  """Generate ItemFeatures:

  img_id, img_width, img_height, question-answers map, img_feats, img_boxes,
  ocr_tokens, ocr_feats, ocr_boxes, ocr_fasttext, ocr_phoc
  """

  def __init__(self, cfg, splits='train'):
    """
        splits: in ("train", "val", "test")
        save_img_cache: whether save image_featrues during loading batches
        :param
        """
    super().__init__()

    # jinliang comment
    # self.vqa_path_config = vqa_path_config
    # self.splits = splits.split(";")
    # self.mmf_features_dir = [vqa_path_config["mmf_features"][split] for split in self.splits]
    # self.mmf_annotations_path = [vqa_path_config["mmf_annotations"][split] for split in self.splits]

    self.splits = splits
    self.mmf_features_dir = []
    self.mmf_annotations_path = []
    for data in self.splits:
      self.mmf_features_dir.append(cfg.mmf_features[data])
      self.mmf_annotations_path.append(cfg.mmf_annotations[data])
    self.mmf_annotations = np.concatenate(
        [np.load(p, allow_pickle=True)[1:] for p in self.mmf_annotations_path])

    # for index, tmp in enumerate(self.mmf_annotations):
    #     if "question_tokens" not in tmp.keys():
    #         print("heihei")

    self.features_pathes = {}
    for split in self.splits:
      mmf_features_dir_tmp = self.mmf_features_dir[self.splits.index(split)]
      names = os.listdir(mmf_features_dir_tmp)
      for name in names:
        self.features_pathes[split + '_' +
                             name.split('.pth')[0]] = os.path.join(
                                 mmf_features_dir_tmp, name)

  def __len__(self):
    return len(self.mmf_annotations)

  def __getitem__(self, item):
    annotation = self.mmf_annotations[item]
    itemFeature = ItemFeature()
    for k, v in annotation.items():
      itemFeature[k] = v

    # TODO(jinliang)
    # itemFeature.tokens = annotation["question_tokens"]
    # itemFeature.answers = annotation["answers"]
    # itemFeature.all_answers = annotation["all_answers"]
    # print(item)
    # itemFeature.ocr_tokens = annotation["ocr_tokens"]

    img_name = annotation['image_name']
    if 'train' in img_name:
      split = 'train'
    elif 'val' in img_name:
      split = 'val'
    elif 'test' in img_name:
      split = 'test'
    else:
      split = 'visualgenome'
    if 'oneval' in self.splits:
      split = 'oneval'

    if split is not 'test':
      itemFeature.answers = annotation['answers']
      itemFeature.all_answers = annotation['all_answers']

    itemFeature.tokens = annotation['question_tokens']
    itemFeature.img_id = annotation['image_id']
    feature_path = self.features_pathes[split + '_' + str(itemFeature.img_id)]
    itemFeature.feature = torch.load(feature_path)[0]
    return itemFeature
