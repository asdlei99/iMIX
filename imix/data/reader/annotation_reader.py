from torch.utils.data import Dataset
import numpy as np


class AnnotationBaseData(Dataset):

    def __init__(self, splits, annotation_cfg, *args, **kwargs):
        super().__init__()

        self.splits = splits
        self.annotation_cfg = annotation_cfg

        self.annotations = []
        self.item_splits = []

        self._load()

    def _load(self):
        for data in self.splits:
            file = self.annotation_cfg[data]
            file_format = file.split('.')[-1]

            assert file_format in ['json', 'npy']

            if file_format == 'npy':
                a = self._load_by_npy(path=file)
            elif file_format == 'json':
                a = self._load_by_json(path=file)

            self.annotations.extend(a)
            self.item_splits.extend([data] * len(a))

    @staticmethod
    def _load_by_json(path):
        return path

    @staticmethod
    def _load_by_npy(path):
        data = np.load(file=path, allow_pickle=True)
        if 'version' in data[0] or 'image_id' not in data[0]:
            return data[1:]
        else:
            return data

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations[idx]
        return data


def build_annotations(splits, annotation_cfg):
    annotation_bd = AnnotationBaseData(splits=splits, annotation_cfg=annotation_cfg)
    item_splits = annotation_bd.item_splits
    return annotation_bd, item_splits

    # self.mix_annotations = []
    # self.item_splits = []
    #
    # for dataset in self.splits:
    #     an = np.load(self.cfg.mix_annotations[dataset], allow_pickle=True)[1:]
    #     self.mix_annotations.extend(an)
    #     self.item_splits.extend([dataset] * len(an))
