# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torch.utils.data import Dataset
from pytorch_transformers.tokenization_bert import BertTokenizer
from .datasets import DatasetMapTrain
from .datasets._image_features_reader import ImageFeaturesH5Reader
from imix.data.builder import DATASETS

logger = logging.getLogger(__name__)


@DATASETS.register_module()
class LoadDatasets(Dataset):

    def __init__(self, reader):
        task_cfg = reader
        tokenizer = BertTokenizer.from_pretrained(task_cfg['bert_model'], do_lower_case=task_cfg['do_lower_case'])

        task_feature_reader1 = {}
        task_feature_reader2 = {}
        self.task = []
        self._limit_sample_nums = task_cfg.get('limit_nums', None)

        ids = task_cfg['tasks'][0].split('-')
        for i, task_id in enumerate(ids):
            task = 'TASK' + task_id
            self.task.append(task)
            if task_cfg[task]['features_h5path1'] not in task_feature_reader1:
                task_feature_reader1[task_cfg[task]['features_h5path1']] = None
            if task_cfg[task]['features_h5path2'] not in task_feature_reader2:
                task_feature_reader2[task_cfg[task]['features_h5path2']] = None

        # initilzie the feature reader
        for features_h5path in task_feature_reader1.keys():
            if features_h5path != '':
                task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(features_h5path, task_cfg['in_memory'])
        for features_h5path in task_feature_reader2.keys():
            if features_h5path != '':
                task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(features_h5path, task_cfg['in_memory'])

        self.task_datasets = {}
        # task_ids = []
        # task_batch_size = {}
        # task_num_iters = {}

        # only one task now
        for i, task_id in enumerate(ids):
            task = 'TASK' + task_id
            task_name = task_cfg[task]['name']
            # task_ids.append(task)
            '''
            batch_size = task_cfg[task]["batch_size"] // task_cfg[
                'gradient_accumulation_steps']

            # num_workers = int(num_workers / len(ids))
            logger.info("Loading %s Dataset with batch size %d" %
                        (task_cfg[task]["name"], batch_size))
            '''

            self.task_datasets[task] = DatasetMapTrain[task_name](
                task=task_cfg[task]['name'],
                dataroot=task_cfg[task]['dataroot'],
                annotations_jsonpath=task_cfg[task]['annotations_jsonpath'],
                split=task_cfg[task]['split'],
                image_features_reader=task_feature_reader1[task_cfg[task]['features_h5path1']],
                gt_image_features_reader=task_feature_reader2[task_cfg[task]['features_h5path2']],
                tokenizer=tokenizer,
                bert_model=task_cfg['bert_model'],
                clean_datasets=task_cfg['clean_datasets'],
                padding_index=0,
                max_seq_length=task_cfg[task]['max_seq_length'],
                max_region_num=task_cfg[task]['max_region_num'])
            # limit_nums=self._limit_sample_nums)

    def __len__(self):
        # only one task now
        return self.task_datasets[self.task[0]].__len__()

    def __getitem__(self, item: int):
        # only one task now
        return self.task_datasets[self.task[0]].__getitem__(item)


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores
