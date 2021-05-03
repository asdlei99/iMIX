import torch
import os
import _pickle as cPickle
from pathlib import Path
from torch import Tensor
import tqdm


def tesnor2list(data):
    new_data = []
    for d in tqdm.tqdm(data):
        tmp = {}
        for k, v in d.items():
            if isinstance(v, Tensor):
                tmp[k] = v.tolist()
            else:
                tmp[k] = v
        new_data.append(tmp)

    return new_data


def save_data(data, file_name):
    with open(file_name, 'wb') as f:
        cPickle.dump(data, f)
    print(file_name)
    print('==' * 20)


vilbert_dataset_root = '/home/datasets/mix_data/vilbert/datasets'
dataset_dir = 'visual7w/cache'

dataset_path = os.path.join(vilbert_dataset_root, dataset_dir)
dataset_path = os.path.normpath(dataset_path)
annotations_files = Path(dataset_path).glob('*cleaned.pkl')
for file in annotations_files:
    with open(file, 'rb') as f:
        data = cPickle.load(f)
        data = tesnor2list(data)
        file = str(file)
        file_name = file.split('.')[0] + '_tolist.' + file.split('.')[-1]
        save_data(data, file_name)
