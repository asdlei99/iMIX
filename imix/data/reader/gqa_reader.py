"""
author: zrz
created time: 2021/1/14
"""

import pickle

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
        item_feature = ItemFeature(annotation)
        item_feature.error = False
        #for k, v in annotation.items():
        #    item_feature[k] = v

        # TODO(jinliang)
        # item_feature.tokens = annotation["question_tokens"]
        # item_feature.answers = annotation["answers"]
        # item_feature.all_answers = annotation["all_answers"]
        # print(item)
        # item_feature.ocr_tokens = annotation["ocr_tokens"]

        #if split is not 'test':
        #    item_feature.answers = annotation['answers']
        #    item_feature.all_answers = annotation['all_answers']

        # bert use self.tokenizer TODO zhangrunze

        item_feature.tokens = tokenize_gqa(annotation['question_str'])
        item_feature.tokens_len = len(item_feature.tokens)

        item_feature.img_id = annotation['image_id']
        if self.default_feature:
            feature_info = None
            for txn in self.feature_txns:
                feature_info = pickle.loads(txn.get(annotation['image_name'].encode()))
                if feature_info is not None:
                    break
            if feature_info is None:
                item_feature.error = True
                item_feature.feature = np.random.random((100, 2048))
                return item_feature
            item_feature.update(feature_info.items())
            #for k, v in feature_info.items():
            #    item_feature[k] = v
            return item_feature

        feature_path = self.features_pathes[split + '_' + str(item_feature.img_id)]
        item_feature.feature = torch.load(feature_path)[0]
        return item_feature
