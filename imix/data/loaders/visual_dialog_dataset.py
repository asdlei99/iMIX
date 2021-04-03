from torch.utils.data import Dataset
from imix.data.reader.visual_dialog_reader import VisDiaReader
from imix.data.infocomp.visual_dialog_infocpler import VisDiaInfoCpler
from imix.data.builder import DATASETS


@DATASETS.register_module()
class VisDialDataset(Dataset):

    def __init__(self, reader, info_cpler, limit_nums=None):
        self.reader = VisDiaReader(reader)
        self.info_cpler = VisDiaInfoCpler(info_cpler)
        self._limit_sample_nums = limit_nums
        self._splits = self.reader.splits

    def __len__(self):
        if self._limit_sample_nums and self._limit_sample_nums > 0:
            return min(len(self.reader), self._limit_sample_nums)
        return len(self.reader)

    def __getitem__(self, idx):
        item_feature = self.reader[idx]
        item = self.info_cpler.complete_info(item_feature=item_feature, split=self._splits[0])
        return item
