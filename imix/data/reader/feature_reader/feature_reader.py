import lmdb
import os
import pickle

from imix.utils_imix.registry import Registry, build_from_cfg

FeatureReaders = Registry('FeatureReaders')


def build_feature_reader(cfg, default_args=None):
    return build_from_cfg(cfg=cfg, registry=FeatureReaders, default_args=default_args)


class FeatureReader:

    def __init__(self, dataset_type, feat_path=None, max_features=None):
        self.dataset_type = dataset_type
        self.feat_path = feat_path
        self.max_features = max_features
        self.feat_reader = None

    def read_by_path(self, img_feat_path):
        pass


@FeatureReaders.register_module()
class LMDBFeatureReader(FeatureReader):

    def __init__(self, dataset_type, feat_path=None, max_features=None):
        super().__init__(dataset_type, feat_path=feat_path, max_features=max_features)
        self.env_db = None

    def _init_env_db(self):
        self.env_db = lmdb.open(
            path=self.feat_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=os.path.isdir(self.feat_path))
        with self.env_db.begin(write=False, buffers=True) as txn:
            keys = txn.get(b'keys')
            if keys is None:
                self.img_ids = None
                self.img_ids_index = None
            else:
                self.img_ids = pickle.loads(keys)
                self.img_ids_index = {self.img_ids[idx]: idx for idx in range(0, len(self.img_ids))}

    def _get_feature(self, key):
        with self.env_db.begin(write=False, buffers=True) as txn:
            try:
                feature = pickle.loads(txn.get(key.encode()))
            except TypeError:
                feature = None

        return feature

    def read(self, img_annotation):
        if self.env_db is None:
            self._init_env_db()

        if self.dataset_type == 'TEXTVQAREADER':  # TODO(jinliang)
            dataset_name = img_annotation.get('set_name', None)
            img_name = img_annotation.get('image_name', None)
            if None in [dataset_name, img_name]:
                return None
            else:
                key = dataset_name + '/' + img_name
        elif self.dataset_type in ['GQAReader', 'VQAReader']:
            img_name = img_annotation.get('image_name', None)
            if img_name is None:
                return None
            else:
                key = img_name

        return self.read_by_path(img_feat_path=key)

    def read_by_path(self, img_feat_path):
        img_feature = self._get_feature(key=img_feat_path)
        return img_feature
