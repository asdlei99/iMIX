"""
author: lxc
created time: 2020/8/18
"""

import numpy as np
import os
import torch
import lmdb
import pickle
from .base_reader import BaseDataReader
from ..utils.stream import ItemFeature
from ..utils.tokenization import BertTokenizer
import json
from lz4.frame import compress, decompress
import horovod.torch as hvd
import msgpack
import msgpack_numpy
from tqdm import tqdm
msgpack_numpy.patch()


# def _check_distributed():
#   try:
#     dist = hvd.size() != hvd.local_size()
#   except ValueError:
#     # not using horovod
#     dist = False
#   return dist
#
# def _fp16_to_fp32(feat_dict):
#   out = {k: arr.astype(np.float32)
#   if arr.dtype == np.float16 else arr
#          for k, arr in feat_dict.items()}
#   return out

def compute_num_bb(confs, conf_th, min_bb, max_bb):
  num_bb = max(min_bb, (confs > conf_th).sum())
  num_bb = min(max_bb, num_bb)
  return num_bb

def _get_vqa_target(example, num_answers):
  target = torch.zeros(num_answers)
  labels = example['target']['labels']
  scores = example['target']['scores']
  if labels and scores:
    target.scatter_(0, torch.tensor(labels), torch.tensor(scores))
  return target


class DetectFeatLmdb(object):
  def __init__(self, img_dir, conf_th=0.2, max_bb=100, min_bb=10, num_bb=36,
               compress=True):
    self.img_dir = img_dir
    if conf_th == -1:
      db_name = f'feat_numbb{num_bb}'
      self.name2nbb = defaultdict(lambda: num_bb)
    else:
      db_name = f'feat_th{conf_th}_max{max_bb}_min{min_bb}'
      nbb = f'nbb_th{conf_th}_max{max_bb}_min{min_bb}.json'
      if not os.path.exists(f'{img_dir}/{nbb}'):
        # nbb is not pre-computed
        self.name2nbb = None
      else:
        self.name2nbb = json.load(open(f'{img_dir}/{nbb}'))
    self.compress = compress
    if compress:
      db_name += '_compressed'

    if self.name2nbb is None:
      if compress:
        db_name = 'all_compressed'
      else:
        db_name = 'all'
    # only read ahead on single node training
    self.env = lmdb.open(f'{img_dir}/{db_name}',
                         readonly=True, create=False,
                         # readahead=not _check_distributed())
                         readahead = True)
    self.txn = self.env.begin(buffers=True)
    if self.name2nbb is None:
      self.name2nbb = self._compute_nbb()

  def _compute_nbb(self):
    name2nbb = {}
    fnames = json.loads(self.txn.get(key=b'__keys__').decode('utf-8'))
    for fname in tqdm(fnames, desc='reading images'):
      dump = self.txn.get(fname.encode('utf-8'))
      if self.compress:
        with io.BytesIO(dump) as reader:
          img_dump = np.load(reader, allow_pickle=True)
          confs = img_dump['conf']
      else:
        img_dump = msgpack.loads(dump, raw=False)
        confs = img_dump['conf']
      name2nbb[fname] = compute_num_bb(confs, self.conf_th,
                                       self.min_bb, self.max_bb)

    return name2nbb

  def __del__(self):
    self.env.close()

  def get_dump(self, file_name):
    # hack for MRC
    dump = self.txn.get(file_name.encode('utf-8'))
    nbb = self.name2nbb[file_name]
    if self.compress:
      with io.BytesIO(dump) as reader:
        img_dump = np.load(reader, allow_pickle=True)
        img_dump = _fp16_to_fp32(img_dump)
    else:
      img_dump = msgpack.loads(dump, raw=False)
      img_dump = _fp16_to_fp32(img_dump)
    img_dump = {k: arr[:nbb, ...] for k, arr in img_dump.items()}
    return img_dump

  def __getitem__(self, file_name):
    dump = self.txn.get(file_name.encode('utf-8'))
    nbb = self.name2nbb[file_name]
    if self.compress:
      with io.BytesIO(dump) as reader:
        img_dump = np.load(reader, allow_pickle=True)
        img_dump = {'features': img_dump['features'],
                    'norm_bb': img_dump['norm_bb']}
    else:
      img_dump = msgpack.loads(dump, raw=False)
    img_feat = torch.tensor(img_dump['features'][:nbb, :]).float()
    img_bb = torch.tensor(img_dump['norm_bb'][:nbb, :]).float()
    return img_feat, img_bb

class TxtLmdb(object):
  def __init__(self, db_dir, readonly=True):
    self.readonly = readonly
    if readonly:
      # training
      self.env = lmdb.open(db_dir,
                           readonly=True, create=False,
                           # readahead=not _check_distributed())
                           readahead = True)
      self.txn = self.env.begin(buffers=True)
      self.write_cnt = None
    else:
      # prepro
      self.env = lmdb.open(db_dir, readonly=False, create=True,
                           map_size=4 * 1024 ** 4)
      self.txn = self.env.begin(write=True)
      self.write_cnt = 0

  def __del__(self):
    if self.write_cnt:
      self.txn.commit()
    self.env.close()

  def __getitem__(self, key):
    return msgpack.loads(decompress(self.txn.get(key.encode('utf-8'))),
                         raw=False)

  def __setitem__(self, key, value):
    # NOTE: not thread safe
    if self.readonly:
      raise ValueError('readonly text DB')
    ret = self.txn.put(key.encode('utf-8'),
                       compress(msgpack.dumps(value, use_bin_type=True)))
    self.write_cnt += 1
    if self.write_cnt % 1000 == 0:
      self.txn.commit()
      self.txn = self.env.begin(write=True)
      self.write_cnt = 0
    return ret


class TxtTokLmdb(object):
  def __init__(self, db_dir, max_txt_len=60):
    if max_txt_len == -1:
      self.id2len = json.load(open(f'{db_dir}/id2len.json'))
    else:
      self.id2len = {
        id_: len_
        for id_, len_ in json.load(open(f'{db_dir}/id2len.json')
                                   ).items()
        if len_ <= max_txt_len
      }
    self.db_dir = db_dir
    self.db = TxtLmdb(db_dir, readonly=True)
    meta = json.load(open(f'{db_dir}/meta.json', 'r'))
    self.cls_ = meta['CLS']
    self.sep = meta['SEP']
    self.mask = meta['MASK']
    self.v_range = meta['v_range']

  def __getitem__(self, id_):
    txt_dump = self.db[id_]
    return txt_dump

  def combine_inputs(self, *inputs):
    input_ids = [self.cls_]
    for ids in inputs:
      input_ids.extend(ids + [self.sep])
    return torch.tensor(input_ids)

  @property
  def txt2img(self):
    txt2img = json.load(open(f'{self.db_dir}/txt2img.json'))
    return txt2img

  @property
  def img2txts(self):
    img2txts = json.load(open(f'{self.db_dir}/img2txts.json'))
    return img2txts


class VQAReaderUNITER(BaseDataReader):

  def __init__(self, cfg):
    super().__init__(cfg)
    self.conf_th = 0.2
    self.max_bb = 100
    self.min_bb = 10
    self.num_bb = 36
    self.compress = False
    self.max_txt_len = 60
    path = cfg.mix_features.train
    txt_path = cfg.mix_annotations.train
    self.img_db = DetectFeatLmdb(path, self.conf_th, self.max_bb,
                                 self.min_bb, self.num_bb, self.compress)
    self.txt_db = TxtTokLmdb(txt_path, self.max_txt_len)
    txt_lens, self.ids = self.get_ids_and_lens(self.txt_db)

    txt2img = self.txt_db.txt2img
    self.lens = [tl + self.img_db.name2nbb[txt2img[id_]]
                 for tl, id_ in zip(txt_lens, self.ids)]

  def __len__(self):
    return len(self.ids)

  def get_ids_and_lens(self, db):
    assert isinstance(db, TxtTokLmdb)
    lens = []
    ids = []
    # for id_ in list(db.id2len.keys())[hvd.rank()::hvd.size()]:
    for id_ in list(db.id2len.keys()):
      lens.append(db.id2len[id_])
      ids.append(id_)
    return lens, ids


  def __getitem__(self, item):

    id_ = self.ids[item]
    example = self.txt_db[id_]
    img_feat, img_pos_feat, num_bb = self._get_img_feat(example['img_fname'])

    # text input
    input_ids = example['input_ids']
    input_ids = self.txt_db.combine_inputs(input_ids)

    target = _get_vqa_target(example, self.num_answers)

    attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
    itemFeature = ItemFeature()

    return input_ids, img_feat, img_pos_feat, attn_masks, target

  def _get_img_feat(self, fname):
      img_feat, bb = self.img_db[fname]
      img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
      num_bb = img_feat.size(0)
      return img_feat, img_bb, num_bb

    # annotation = self.mix_annotations[item]
    # split = self.item_splits[item]
    # itemFeature = ItemFeature()
    # itemFeature.error = False
    # for k, v in annotation.items():
    #   itemFeature[k] = v
    #
    # # TODO(jinliang)
    # # itemFeature.tokens = annotation["question_tokens"]
    # # itemFeature.answers = annotation["answers"]
    # # itemFeature.all_answers = annotation["all_answers"]
    # # print(item)
    # # itemFeature.ocr_tokens = annotation["ocr_tokens"]
    #
    # if split is not 'test':
    #   itemFeature.answers = annotation['answers']
    #   itemFeature.all_answers = annotation['all_answers']
    #
    #
    # itemFeature.tokens = annotation['question_tokens']
    # itemFeature.img_id = annotation['image_id']
    # if self.default_feature:
    #   feature_info = None
    #   for txn in self.feature_txns:
    #     feature_info = pickle.loads(txn.get(annotation['image_name'].encode()))
    #     if feature_info is not None:
    #       break
    #   feature_global_info = None
    #   for txn in self.feature_global_txns:
    #     feature_global_info = pickle.loads(txn.get(annotation['image_name'].encode()))
    #     if feature_global_info is None:
    #       break
    #     else:
    #       feature_global_info['global_feature_path'] = feature_global_info.pop('feature_path')
    #       feature_global_info['global_features'] = feature_global_info.pop('features')
    #   if feature_info is None or feature_global_info is None:
    #     itemFeature.error = True
    #     itemFeature.feature = np.random.random((100, 2048))
    #     itemFeature.global_feature = np.random.random((100, 2048))
    #     return itemFeature
    #
    #   for k, v in feature_info.items():
    #     itemFeature[k] = v
    #   for k, v in feature_global_info.items():
    #     itemFeature[k] = v
    #   return itemFeature
    # feature_path = self.features_pathes[split + '_' + str(itemFeature.img_id)]
    # itemFeature.feature = torch.load(feature_path)[0]
    # return itemFeature
