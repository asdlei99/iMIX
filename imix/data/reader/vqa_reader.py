"""
author: lxc
created time: 2020/8/18
"""

import numpy as np
import os
import torch
import lmdb
import pickle
from .base_reader import IMIXDataReader
from ..utils.stream import ItemFeature
from ..utils.tokenization import BertTokenizer
from .base_reader import BaseDataReader
from ..utils.io import data_dump


class VQAReader(IMIXDataReader):

    def __init__(self, cfg):
        super().__init__(cfg)

    def __len__(self):
        return len(self.mix_annotations)

    def __getitem__(self, item):
        annotation = self.mix_annotations[item]
        split = self.item_splits[item]
        item_feature = ItemFeature(annotation)
        #item_feature = ItemFeature()

        #for k, v in annotation.items():
        #    item_feature[k] = v

        item_feature.error = False
        # TODO(jinliang)
        # item_feature.tokens = annotation["question_tokens"]
        # item_feature.answers = annotation["answers"]
        # item_feature.all_answers = annotation["all_answers"]
        # print(item)
        # item_feature.ocr_tokens = annotation["ocr_tokens"]

        if split is not 'test':
            item_feature.answers = annotation['answers']
            item_feature.all_answers = annotation['all_answers']

        item_feature.tokens = annotation['question_tokens']
        item_feature.img_id = annotation['image_id']
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

            if self.if_global:
                if feature_info is None or feature_global_info is None:
                    item_feature.error = True
                    item_feature.feature = np.random.random((100, 2048))
                    item_feature.global_feature = np.random.random((100, 2048))
                    return item_feature
            else:
                if feature_info is None:
                    item_feature.error = True
                    item_feature.feature = np.random.random((100, 2048))
                    return item_feature

            for k, v in feature_info.items():
                item_feature[k] = v
            if self.if_global:
                for k, v in feature_global_info.items():
                    item_feature[k] = v
            return item_feature
        feature_path = self.features_pathes[split + '_' + str(item_feature.img_id)]
        item_feature.feature = torch.load(feature_path)[0]
        return item_feature
