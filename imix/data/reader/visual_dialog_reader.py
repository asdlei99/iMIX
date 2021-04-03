import numpy as np
import os
import json
import lmdb
from ..utils.stream import ItemFeature
from ..utils.image_features_reader import ImageFeaturesH5Reader
from ..utils.data_utils import encode_image_input
from typing import Dict


def load_annotation_file(path):
    if path.endswith('.npy'):
        return np.load(path, allow_pickle=True)[1:]


def load_dense_file(path):
    if path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)


class BaseDatasetReader:

    def __init__(self, cfg):
        self.init_default_params(cfg)
        self.has_global = cfg.get('image_global_features') is not None
        self.has_ocr = cfg.get('ocr_features') is not None
        self.image_feature_max_regions = cfg.get('image_feature_max_regions', 37)

        self.image_features_dir = []
        self.image_global_features_dir = []
        self.ocr_features_dir = []
        self.annotations_path = []
        self.annotations = []
        self.item_splits = []

        self.if_global = cfg.if_global if self.has_global else False
        self.load_data_from_cfg(cfg)

    def init_default_params(self, cfg):
        self.card = cfg.get('card', 'default')
        assert self.card in ['default', 'grid']
        self.default_feature = self.card == 'default'
        splits = cfg.datasets
        if isinstance(splits, str):
            splits = [splits]
        self.splits = splits

    def decode_val_annoation(self, cfg_annotations: Dict) -> list:
        val_an = load_annotation_file(cfg_annotations['val'])
        dense_an = load_dense_file(cfg_annotations['dense'])
        assert len(val_an) == len(dense_an)
        annoation = []
        for va, da in zip(val_an, dense_an):
            a = dict(va)
            a.update(da)
            annoation.append(a)
        return annoation

    def load_annotation(self, split: str, cfg_an: Dict) -> list:
        if split == 'train':
            return load_annotation_file(cfg_an[split])
        elif split == 'val':
            return self.decode_val_annoation(cfg_an)
        elif split == 'test':
            pass

    def load_data_from_cfg(self, cfg):
        for data in self.splits:
            self.image_features_dir.append(cfg.image_features[data])
            self.annotations_path.append(cfg.annotations[data])

            annotation = self.load_annotation(split=data, cfg_an=cfg.annotations)
            self.annotations.extend(annotation)
            self.item_splits.extend([data] * len(annotation))

            if self.has_global and self.has_global:
                self.image_global_features_dir.append(cfg.image_global_features[data])
            if self.has_ocr:
                self.ocr_features_dir.append(cfg.ocr_features[data])

        if self.default_feature:
            self.feature_txns = []
            self.feature_global_txns = []
            self.feature_ocr_txns = []
            for path in set(self.image_features_dir):
                self.feature_txns.append(lmdb.open(path).begin())
            for path in set(self.image_global_features_dir):
                self.feature_global_txns.append(lmdb.open(path).begin())
            for path in set(self.ocr_features_dir):
                self.feature_ocr_txns.append(lmdb.open(path).begin())
        else:
            self.features_pathes = {}
            for split in self.splits:
                img_feature_dir = self.image_features_dir[self.splits.index(split)]
                names = os.listdir(img_feature_dir)
                for name in names:
                    self.features_pathes[split + '_' + name.split('.pth')[0]] = os.path.join(img_feature_dir, name)


class VisDiaReader(BaseDatasetReader):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.image_features_reader = ImageFeaturesH5Reader(
            self.image_features_dir[0])  # TODO(jinliang)  temporary treatement

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annoation = self.annotations[idx]
        # split = self.item_splits[idx]
        item_feature = ItemFeature(init_dict=annoation)
        item_feature.error = False
        item_feature.update(self.get_image_feature_info(annoation['image_id']))

        return item_feature

    def get_image_feature_info(self, idx):
        features, num_boxes, boxes, _, image_target = self.image_features_reader[idx]
        features, spatials, image_mask, image_target, image_label = encode_image_input(
            features, num_boxes, boxes, image_target, max_regions=self.image_feature_max_regions)
        image_feature_info = {}
        image_feature_info['image_feat'] = features
        image_feature_info['image_loc'] = spatials
        image_feature_info['image_mask'] = image_mask
        image_feature_info['image_target'] = image_target
        image_feature_info['image_label'] = image_label

        return image_feature_info
