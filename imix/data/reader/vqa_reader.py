"""
author: lxc
created time: 2020/8/18
"""

import numpy as np
from ..utils.stream import ItemFeature
from .base_reader import IMIXDataReader
from imix.utils_imix.common_function import update_d1_with_d2


class VQAReader(IMIXDataReader):

    def __init__(self, cfg):
        super().__init__(cfg)

    def __len__(self):
        return len(self.mix_annotations)

    # def __getitem__(self, item):
    #     annotation = self.mix_annotations[item]
    #     split = self.item_splits[item]
    #     item_feature = ItemFeature(annotation)
    #     '''
    #     item_feature = ItemFeature()
    #
    #     for k, v in annotation.items():
    #         item_feature[k] = v
    #     '''
    #     item_feature.error = False
    #     # TODO(jinliang)
    #     # item_feature.tokens = annotation["question_tokens"]
    #     # item_feature.answers = annotation["answers"]
    #     # item_feature.all_answers = annotation["all_answers"]
    #     # print(item)
    #     # item_feature.ocr_tokens = annotation["ocr_tokens"]
    #
    #     # if split != 'test':
    #     #     item_feature.answers = annotation['answers']
    #     #     item_feature.all_answers = annotation['all_answers']
    #
    #     item_feature.tokens = annotation['question_tokens']
    #     item_feature.img_id = annotation['image_id']
    #     if self.default_feature:
    #         feature_info = None
    #         for txn in self.feature_txns:
    #             feature_info = pickle.loads(txn.get(annotation['image_name'].encode()))
    #             if feature_info is not None:
    #                 break
    #         feature_global_info = None
    #         for txn in self.feature_global_txns:
    #             feature_global_info = pickle.loads(txn.get(annotation['image_name'].encode()))
    #             if feature_global_info is None:
    #                 break
    #             else:
    #                 feature_global_info['global_feature_path'] = feature_global_info.pop('feature_path')
    #                 feature_global_info['global_features'] = feature_global_info.pop('features')
    #
    #         if self.is_global:
    #             if feature_info is None or feature_global_info is None:
    #                 item_feature.error = True
    #                 item_feature.features = np.random.random((100, 2048))
    #                 item_feature.global_feature = np.random.random((100, 2048))
    #                 return item_feature
    #         else:
    #             if feature_info is None:
    #                 item_feature.error = True
    #                 item_feature.features = np.random.random((100, 2048))
    #                 return item_feature
    #
    #         for k, v in feature_info.items():
    #             item_feature[k] = v
    #         if self.is_global:
    #             for k, v in feature_global_info.items():
    #                 item_feature[k] = v
    #         return item_feature
    #     feature_path = self.features_pathes[split + '_' + str(item_feature.img_id)]
    #     item_feature.features = torch.load(feature_path)[0]
    #     return item_feature

    def __getitem__(self, idx):
        annotation = self.mix_annotations[idx]

        item_feature = ItemFeature(annotation)
        item_feature.error = False
        item_feature.tokens = annotation['question_tokens']
        item_feature.img_id = annotation['image_id']

        feature = self.feature_obj[idx]
        global_feature = None

        if self.global_feature_obj:
            global_feature = self.global_feature_obj[idx]
            global_feature['global_features'] = global_feature.pop('features')
            global_feature['global_feature_path'] = global_feature.pop('feature_path')

        if self.is_global:
            if None in [feature, global_feature]:
                item_feature.error = True
                item_feature.features = np.random.random((100, 2048))
                item_feature.global_feature = np.random.random((100, 2048))
                return item_feature
        else:
            if feature is None:
                item_feature.error = True
                item_feature.features = np.random.random((100, 2048))
                return item_feature

        update_d1_with_d2(d1=item_feature, d2=feature)
        if self.is_global:
            update_d1_with_d2(d1=item_feature, d2=global_feature)

        return item_feature
