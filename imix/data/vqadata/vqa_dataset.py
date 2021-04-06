import logging

from torch.utils.data import Dataset

import imix.utils_mix.distributed_info as comm
from ..builder import DATASETS
from ..infocomp.vqa_infocpler import VQAInfoCpler
from ..reader.vqa_reader import VQAReader

# VQA_PATH_CONFIG = yaml.load(open("datasets/dataset_vqa.yaml"))["dataset_configs"]

# @DATASETS.register_module()
# class VQADATASET(Dataset):
#     def __init__(self, splits):
#         self.reader = VQAReader(VQA_PATH_CONFIG, splits)
#         self.infocpler = VQAInfoCpler(VQA_PATH_CONFIG)
#
#     def __len__(self):
#         return len(self.reader)
#
#     def __getitem__(self, item):
#
#         itemFeature = self.reader[item]
#         itemFeature = self.infocpler.completeInfo(itemFeature)
#         return itemFeature.feature, itemFeature.input_ids, itemFeature.answers_scores, itemFeature.input_mask


@DATASETS.register_module()
class VQADATASET(Dataset):

    def __init__(self, vqa_reader, vqa_info_cpler, limit_nums=None):
        if comm.is_main_process():
            logger = logging.getLogger(__name__)
            logger.info('start loading vqadata')

        self.reader = VQAReader(vqa_reader, vqa_reader.datasets)
        self.infocpler = VQAInfoCpler(vqa_info_cpler)
        self._limit_sample_nums = limit_nums
        self._split = vqa_reader.datasets
        if comm.is_main_process():
            logger.info('load vqadata {} successfully'.format(vqa_reader.datasets))

    def __len__(self):
        if self._limit_sample_nums and self._limit_sample_nums > 0:
            return min(len(self.reader), self._limit_sample_nums)
        return len(self.reader)

    def __getitem__(self, idx):
        itemFeature = self.reader[idx]
        itemFeature = self.infocpler.completeInfo(itemFeature)

        # TODO(jinliang)
        item = {
            'feature': itemFeature.feature,
            'input_ids': itemFeature.input_ids,
            'input_mask': itemFeature.input_mask,
            'question_id': itemFeature.question_id,
            'image_id': itemFeature.image_id
        }

        if itemFeature.answers_scores is not None:
            item['answers_scores'] = itemFeature.answers_scores
        # return itemFeature.feature, itemFeature.input_ids, itemFeature.answers_scores, itemFeature.input_mask

        if 'test' in self._split or 'oneval' in self._split:
            item['quesid2ans'] = self.infocpler.qa_id2ans
        return item

        # return itemFeature
