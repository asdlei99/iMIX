"""
author: zrz
created time: 2021/1/14
"""

import os
import pickle

import lmdb
import numpy as np
import torch

from ..utils.stream import ItemFeature
from ..utils.tokenization import BertTokenizer
from .base_reader import IMIXDataReader


def tokenize_gqa(sentence, ignoredPunct=['?', '!', '\\', '/', ')', '('], keptPunct=['.', ',', ';', ':']):
    sentence = sentence.lower()
    for punct in keptPunct:
        sentence = sentence.replace(punct, ' ' + punct + ' ')
    for punct in ignoredPunct:
        sentence = sentence.replace(punct, '')
    tokens = sentence.split()
    return tokens


class GQAReader(IMIXDataReader):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

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

        if split != 'test':
            itemFeature.answers = annotation['answers']
            itemFeature.all_answers = annotation['all_answers']

        # bert use self.tokenizer TODO zhangrunze

        itemFeature.tokens = tokenize_gqa(annotation['question_str'])
        itemFeature.tokens_len = len(itemFeature.tokens)

        itemFeature.img_id = annotation['image_id']
        if self.default_feature:
            feature_info = None
            for txn in self.feature_txns:
                feature_info = pickle.loads(txn.get(annotation['image_name'].encode()))
                if feature_info is not None:
                    break
            if feature_info is None:
                itemFeature.error = True
                itemFeature.feature = np.random.random((100, 2048))
                return itemFeature
            for k, v in feature_info.items():
                itemFeature[k] = v
            return itemFeature

        feature_path = self.features_pathes[split + '_' + str(itemFeature.img_id)]
        itemFeature.feature = torch.load(feature_path)[0]
        return itemFeature
