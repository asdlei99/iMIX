from torch.utils.data import Dataset
from ..reader.lxmertpretrain_reader import LXMERTPretrainReader as Reader
from ..infocomp.lxmertpretrain_infocpler import LXMERTPreTrainInfoCpler as InfoCpler
from ..builder import DATASETS
from .base_loader import BaseLoader

import logging
import imix.utils_imix.distributed_info as comm


@DATASETS.register_module()
class LXMERTPreTrainDATASET(BaseLoader):

    def __init__(self, reader, info_cpler, limit_nums=None):
        super().__init__(Reader, reader, InfoCpler, info_cpler, limit_nums)
        #if comm.is_main_process():
        #    logger = logging.getLogger(__name__)
        #    logger.info('start loading vqadata')

        #self.reader = LXMERTPretrainReader(reader)
        #self.infocpler = LXMERTPreTrainInfoCpler(info_cpler)
        #self._limit_sample_nums = limit_nums
        #self.splits = self.reader.datasets
        #if comm.is_main_process():
        #    logger.info('load data {} successfully'.format(self.reader.datasets))

    #def __len__(self):
    #    if self._limit_sample_nums and self._limit_sample_nums > 0:
    #        return min(len(self.reader), self._limit_sample_nums)
    #    return len(self.reader)

    def __getitem__(self, idx):
        # idx = 0
        item_feature = self.reader[idx]
        item_feature = self.infocpler.completeInfo(item_feature)

        return item_feature
