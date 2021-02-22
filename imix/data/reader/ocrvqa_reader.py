"""
author: lxc
created time: 2020/8/19
"""

import numpy as np
import os
import torch
import lmdb
import pickle
from .base_reader import MMFDataReader
from ..utils.stream import ItemFeature


class OCRVQAReader(MMFDataReader):

    def __init__(self, cfg):
        super().__init__(cfg)
        assert self.default_feature, ('Not support non-default features now.')

    def __len__(self):
        return len(self.mix_annotations)

    def __getitem__(self, item):
        annotation = self.mix_annotations[item]
        split = self.item_splits[item]
        item_feature = ItemFeature()
        item_feature.error = False
        for k, v in annotation.items():
            item_feature[k] = v

        if split != 'test':
            item_feature.answers = annotation['answers']

        item_feature.tokens = annotation['question_tokens']
        item_feature.img_id = annotation['image_id']

        feature_info = self.get_featureinfo_from_txns(
            self.feature_txns,
            annotation['set_name'] + '/' + annotation['image_name'])
        for k, v in feature_info.items():
            item_feature[k] = v if item_feature.get(
                k) is None else item_feature[k]

        feature_ocr_info = self.get_featureinfo_from_txns(
            self.feature_ocr_txns,
            annotation['set_name'] + '/' + annotation['image_name'])
        feature_ocr_info['features_ocr'] = feature_ocr_info.pop('features')
        for k, v in feature_ocr_info.items():
            item_feature[k] = v if item_feature.get(
                k) is None else item_feature[k]
        item_feature.error = None in [feature_info, feature_ocr_info]

        return item_feature
