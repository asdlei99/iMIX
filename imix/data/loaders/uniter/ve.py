"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Visual entailment dataset # NOTE: basically reuse VQA dataset
"""
from .vqa import VqaDataset, VqaTrainDataset, VqaEvalDataset, vqa_collate, vqa_eval_collate
from ...builder import DATASETS
import imix.utils_imix.distributed_info as comm
import logging
from .data import (TxtTokLmdb, DetectFeatLmdb)


def create_dataloader(img_path, txt_path, is_train, dset_cls, opts):
    img_db = DetectFeatLmdb(img_path, opts['conf_th'], opts['max_bb'], opts['min_bb'], opts['num_bb'], False)
    txt_db = TxtTokLmdb(txt_path, opts['max_txt_len'] if is_train else -1)
    return dset_cls(txt_db, img_db)


@DATASETS.register_module()
class VeDataset(VqaDataset):

    def __init__(self, **kwargs):
        if comm.is_main_process():
            cls_name = self.__class__.__name__
            logger = logging.getLogger(__name__)
            logger.info('start loading' + cls_name)

        # load DBs and image dirs
        opts = kwargs['datacfg'].copy()
        train_or_val = kwargs['train_or_val']
        assert train_or_val is not None
        if train_or_val:  # train
            self.dataset = create_dataloader(opts['train_img_db'], opts['train_txt_db'], True, VeTrainDataset, opts)
        else:
            self.dataset = create_dataloader(opts['train_img_db'], opts['train_txt_db'], True, VeEvalDataset, opts)

        self.collate_fn = self.dataset.collate_fn
        if comm.is_main_process():
            logger.info('load {} successfully'.format(cls_name))


@DATASETS.register_module()
class VeTrainDataset(VqaTrainDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)
        self.collate_fn = ve_collate


@DATASETS.register_module()
class VeEvalDataset(VqaEvalDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)
        self.collate_fn = ve_eval_collate


ve_collate = vqa_collate
ve_eval_collate = vqa_eval_collate
