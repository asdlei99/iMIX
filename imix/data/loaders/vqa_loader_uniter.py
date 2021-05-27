import logging

from torch.utils.data import Dataset

import imix.utils.distributed_info as comm
from ..builder import DATASETS
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

    def __len__(self):
        return len(self.reader.ids)

    def __getitem__(self, idx):

        item_feature = self.reader[idx]

        item = {
            'feature': item_feature.features,  # feature - feature
            'feature_global': item_feature.global_features,  # feature - global_features
            'cls_prob': item_feature.cls_prob,  # 1601 cls_prob
            'bbox': item_feature.bbox,  # feature - bbox
            'image_dim': item_feature.num_boxes,  # feature - bbox_Num
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
