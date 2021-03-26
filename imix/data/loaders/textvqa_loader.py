"""
author: lxc
created time: 2021/1/14
"""

from torch.utils.data import Dataset, IterableDataset
import logging
from ..reader.textvqa_reader import TextVQAReader as Reader
from ..infocomp.textvqa_infocpler import TextVQAInfoCpler as InfoCpler
from ..builder import DATASETS
import imix.utils_imix.distributed_info as comm
from .base_loader import BaseLoader


@DATASETS.register_module()
class TEXTVQADATASET(BaseLoader):

    def __init__(self, reader, info_cpler, limit_nums=None):
        super().__init__(Reader, reader, InfoCpler, info_cpler, limit_nums)
        #if comm.is_main_process():
        #    logger = logging.getLogger(__name__)
        #    logger.info('start loading vqadata')

        #self.reader = Reader(reader)
        #self.infocpler = InfoCpler(info_cpler)
        #self._limit_sample_nums = limit_nums
        #self.splits = reader.datasets
        #if comm.is_main_process():
        #    logger.info('load data {} successfully'.format(reader.datasets))

    #def __len__(self):
    #    if self._limit_sample_nums and self._limit_sample_nums > 0:
    #        return min(len(self.reader), self._limit_sample_nums)
    #    return len(self.reader)

    def __getitem__(self, idx):
        idx = 0
        itemFeature = self.reader[idx]
        itemFeature = self.infocpler.completeInfo(itemFeature)

        if self.infocpler.if_bert:
            # TODO(jinliang+ce@lxc)
            item = {
                'feature': itemFeature.features,  # feature - feature
                'bbox': itemFeature.bbox,  # feature - bbox
                'bbox_normalized': itemFeature.bbox_normalized,
                'feature_global': itemFeature.features_global,
                'feature_ocr': itemFeature.features_ocr,
                'bbox_ocr': itemFeature.bbox_ocr,
                'bbox_ocr_normalized': itemFeature.bbox_ocr_normalized,
                'ocr_vectors_glove': itemFeature.ocr_vectors_glove,
                'ocr_vectors_fasttext': itemFeature.ocr_vectors_fasttext,
                'ocr_vectors_phoc': itemFeature.ocr_vectors_phoc,
                'ocr_vectors_order': itemFeature.ocr_vectors_order,
                'input_ids': itemFeature.input_ids,  # tokens - ids
                'input_mask': itemFeature.input_mask,  # tokens - mask
                'input_segment': itemFeature.input_segment,  # tokens - segments
                'input_lm_label_ids':
                    itemFeature.input_lm_label_ids,  # tokens - mlm labels
                'question_id': itemFeature.question_id,
                'image_id': itemFeature.image_id,
                'train_prev_inds': itemFeature.train_prev_inds,
                'train_loss_mask': itemFeature.train_loss_mask,
            }
        else:
            # TODO(jinliang+ce@lxc)
            item = {
                'feature': itemFeature.features,  # feature - feature
                'bbox': itemFeature.bbox,  # feature - bbox
                'bbox_normalized': itemFeature.bbox_normalized,
                'feature_global': itemFeature.features_global,
                'feature_ocr': itemFeature.features_ocr,
                'bbox_ocr': itemFeature.bbox_ocr,
                'bbox_ocr_normalized': itemFeature.bbox_ocr_normalized,
                'ocr_vectors_glove': itemFeature.ocr_vectors_glove,
                'ocr_vectors_fasttext': itemFeature.ocr_vectors_fasttext,
                'ocr_vectors_phoc': itemFeature.ocr_vectors_phoc,
                'ocr_vectors_order': itemFeature.ocr_vectors_order,
                'input_ids': itemFeature.input_ids,  # tokens - ids
                'input_mask': itemFeature.input_mask,  # tokens - mask
                'question_id': itemFeature.question_id,
                'image_id': itemFeature.image_id,
                'train_prev_inds': itemFeature.train_prev_inds,
                'train_loss_mask': itemFeature.train_loss_mask,
            }

        if itemFeature.answers_scores is not None:
            item['answers_scores'] = itemFeature.answers_scores

        if 'test' in self.splits or 'oneval' in self.splits:
            item['quesid2ans'] = self.infocpler.qa_id2ans
        return item
