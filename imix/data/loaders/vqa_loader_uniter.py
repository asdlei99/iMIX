import logging

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, IterableDataset

# import imix.utils.comm as comm
import imix.utils_imix.distributed_info as comm
from ..builder import DATASETS
from ..infocomp.vqa_infocpler import VQAInfoCpler
from ..reader.vqa_reader import VQAReader
from ..reader.vqa_reader_uniter import VQAReaderUNITER


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
class VQADATASETUNITER(Dataset):

    def __init__(self, reader, info_cpler, limit_nums=None):
        if comm.is_main_process():
            logger = logging.getLogger(__name__)
            logger.info('start loading vqadata')

        self.reader = VQAReaderUNITER(reader)
        # self.infocpler = VQAInfoCpler(info_cpler)
        # self._limit_sample_nums = limit_nums
        # self.splits = reader.datasets
        # if comm.is_main_process():
        #   logger.info('load vqadata {} successfully'.format(reader.datasets))

    def __len__(self):
        return len(self.reader.ids)

    def __getitem__(self, idx):

        itemFeature = self.reader.__getitem__(idx)

        # TODO(jinliang+ce@lxc)
        item = {
            'feature': itemFeature.features,  # feature - feature
            'feature_global': itemFeature.global_features,  # feature - global_features
            'cls_prob': itemFeature.cls_prob,  # 1601 cls_prob
            'bbox': itemFeature.bbox,  # feature - bbox
            'image_dim': itemFeature.num_boxes,  # feature - bbox_Num
            'input_ids': itemFeature.input_ids,  # tokens - ids
            'input_mask': itemFeature.input_mask,  # tokens - mask
            'input_segment': itemFeature.input_segment,  # tokens - segments
            'input_lm_label_ids': itemFeature.input_lm_label_ids,  # tokens - mlm labels
            'question_id': itemFeature.question_id,
            'image_id': itemFeature.image_id,
        }

        if itemFeature.answers_scores is not None:
            item['answers_scores'] = itemFeature.answers_scores

        if 'test' in self.splits or 'oneval' in self.splits:
            item['quesid2ans'] = self.infocpler.qa_id2ans

        return item
