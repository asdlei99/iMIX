"""
author: lxc
created time: 2021/1/14
"""
from ..reader import OCRVQAReader as Reader
from ..infocomp import OCRVQAInfoCpler as InfoCpler
from ..builder import DATASETS
from .base_loader import BaseLoader


@DATASETS.register_module()
class OCRVQADATASET(BaseLoader):

    def __init__(self, reader, info_cpler, limit_nums=None):
        super().__init__(Reader, reader, InfoCpler, info_cpler, limit_nums)
        '''
        if comm.is_main_process():
            logger = logging.getLogger(__name__)
            logger.info('start loading vqadata')

        self.reader = Reader(reader)
        self.infocpler = InfoCpler(info_cpler)
        self._limit_sample_nums = limit_nums
        self.splits = reader.datasets
        if comm.is_main_process():
            logger.info('load data {} successfully'.format(reader.datasets))
        '''

    '''
     def __len__(self):
        if self._limit_sample_nums and self._limit_sample_nums > 0:
            return min(len(self.reader), self._limit_sample_nums)
        return len(self.reader)
    '''

    def __getitem__(self, idx):
        item_feature = self.reader[idx]
        item_feature = self.infocpler.complete_info(item_feature)

        # TODO(jinliang+ce@lxc)
        item = {
            'feature': item_feature.features,  # feature - feature
            'bbox': item_feature.bbox,  # feature - bbox
            'bbox_normalized': item_feature.bbox_normalized,
            'feature_global': item_feature.features_global,
            'feature_ocr': item_feature.features_ocr,
            'bbox_ocr': item_feature.bbox_ocr,
            'bbox_ocr_normalized': item_feature.bbox_ocr_normalized,
            'ocr_vectors_glove': item_feature.ocr_vectors_glove,
            'ocr_vectors_fasttext': item_feature.ocr_vectors_fasttext,
            'ocr_vectors_phoc': item_feature.ocr_vectors_phoc,
            'ocr_vectors_order': item_feature.ocr_vectors_order,
            'input_ids': item_feature.input_ids,  # tokens - ids
            'input_mask': item_feature.input_mask,  # tokens - mask
            'input_segment': item_feature.input_segment,  # tokens - segments
            'input_lm_label_ids': item_feature.input_lm_label_ids,  # tokens - mlm labels
            'question_id': item_feature.question_id,
            'image_id': item_feature.image_id,
        }

        if item_feature.answers_scores is not None:
            item['answers_scores'] = item_feature.answers_scores

        if 'test' in self.splits or 'oneval' in self.splits:
            item['quesid2ans'] = self.infocpler.qa_id2ans
        return item
