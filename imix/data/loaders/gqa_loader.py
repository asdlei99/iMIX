from ..reader.gqa_reader import GQAReader as Reader
from ..infocomp.gqa_infocpler import GQAInfoCpler as InfoCpler
from ..builder import DATASETS
import numpy as np
import torch

import logging

import torch
from torch.utils.data import Dataset

import imix.utils_imix.distributed_info as comm
from .base_loader import BaseLoader


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
class GQADATASET(BaseLoader):

    def __init__(self, reader, info_cpler, limit_nums=None):
        super().__init__(Reader, reader, InfoCpler, info_cpler, limit_nums)
        #if comm.is_main_process():
        #    logger = logging.getLogger(__name__)
        #    logger.info('start loading vqadata')

        #self.reader = GQAReader(reader)
        #self.infocpler = GQAInfoCpler(info_cpler)
        #self._limit_sample_nums = limit_nums
        #self.splits = reader.datasets
        #if comm.is_main_process():
        #    logger.info('load vqadata {} successfully'.format(reader.datasets))

    #def __len__(self):
    #    if self._limit_sample_nums and self._limit_sample_nums > 0:
    #        return min(len(self.reader), self._limit_sample_nums)
    #    return len(self.reader)

    def __getitem__(self, idx):
        # idx = 0
        itemFeature = self.reader[idx]
        itemFeature = self.infocpler.completeInfo(itemFeature)

        # Only test for GQA LCGN ########## TODO zhangrunze
        feature = torch.zeros([36, 2048], dtype=torch.float)
        bbox = torch.zeros([36, 4], dtype=torch.float)
        for idx in range(itemFeature.features.shape[0]):
            bbox[idx] = torch.tensor(itemFeature.bbox[idx])
            feature[idx] = torch.tensor(itemFeature.features[idx])
        itemFeature.bbox = bbox
        itemFeature.features = feature
        ###################################################

        # TODO(jinliang+ce@lxc)
        item = {
            'feature': itemFeature.features,  # feature - feature
            'cls_prob': itemFeature.cls_prob,  # 1601 cls_prob
            'bbox': itemFeature.bbox,  # feature - bbox
            'image_dim': itemFeature.num_boxes,  # feature - bbox_Num
            'input_ids': itemFeature.input_ids,  # tokens - ids
            'questionLengths': itemFeature.tokens_len,
            'input_mask': itemFeature.input_mask,  # tokens - mask
            'input_segment': itemFeature.input_segment,  # tokens - segments
            'input_lm_label_ids': itemFeature.input_lm_label_ids,  # tokens - mlm labels
            'question_id': itemFeature.question_id,
            'image_id': itemFeature.image_id,
        }

        if itemFeature.answers_scores is not None:
            item['answers_scores'] = itemFeature.answers_scores
        # return itemFeature.feature, itemFeature.input_ids, itemFeature.answers_scores, itemFeature.input_mask

        if 'test' in self.splits or 'oneval' in self.splits:
            item['quesid2ans'] = self.infocpler.qa_id2ans

        return remove_None_value_elements(item)

        # return itemFeature
