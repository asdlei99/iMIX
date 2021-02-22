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


class VizWizReader(MMFDataReader):

    def __init__(self, cfg):
        super().__init__(cfg)

    def __len__(self):
        return len(self.mix_annotations)

    def __getitem__(self, item):
        annotation = self.mix_annotations[item]
        split = self.item_splits[item]
        item_feature = ItemFeature()
        item_feature.error = False
        for k, v in annotation.items():
            item_feature[k] = v

        # TODO(jinliang)
        # item_feature.tokens = annotation["question_tokens"]
        # item_feature.answers = annotation["answers"]
        # item_feature.all_answers = annotation["all_answers"]
        # print(item)
        # item_feature.ocr_tokens = annotation["ocr_tokens"]

        if split != 'test':
            item_feature.answers = annotation.get('answers')
            item_feature.all_answers = annotation.get('all_answers')

        item_feature.tokens = annotation.get('question_tokens')
        item_feature.img_id = annotation.get('image_id')
        if self.default_feature:
            feature_info = self.get_featureinfo_from_txns(
                self.feature_txns, annotation.get('image_id'))
            if feature_info is None:
                item_feature.error = True
                item_feature.feature = np.random.random((100, 2048))
                return item_feature
            for k, v in feature_info.items():
                item_feature[k] = v if item_feature.get(
                    k) is None else item_feature[k]
            return item_feature
        feature_path = self.features_pathes[split + '_' +
                                            str(item_feature.img_id)]
        item_feature.feature = torch.load(feature_path)[0]
        return item_feature
