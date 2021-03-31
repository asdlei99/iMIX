import logging

import yaml
from torch.utils.data import Dataset, IterableDataset

import imix.utils_imix.distributed_info as comm
from ..builder import DATASETS
from ..infocomp.lxmertpretrain_infocpler import LXMERTPreTrainInfoCpler
from ..reader.lxmertpretrain_reader import LXMERTPretrainReader


@DATASETS.register_module()
class LXMERTPreTrainDATASET(Dataset):

    def __init__(self, reader, info_cpler, limit_nums=None):
        if comm.is_main_process():
            logger = logging.getLogger(__name__)
            logger.info('start loading vqadata')

        self.reader = LXMERTPretrainReader(reader)
        self.infocpler = LXMERTPreTrainInfoCpler(info_cpler)
        self._limit_sample_nums = limit_nums
        self.splits = self.reader.datasets
        if comm.is_main_process():
            logger.info('load data {} successfully'.format(self.reader.datasets))

    def __len__(self):
        if self._limit_sample_nums and self._limit_sample_nums > 0:
            return min(len(self.reader), self._limit_sample_nums)
        return len(self.reader)

    def __getitem__(self, idx):
        # idx = 0
        item_feature = self.reader[idx]
        item_feature = self.infocpler.completeInfo(item_feature)

        return item_feature
