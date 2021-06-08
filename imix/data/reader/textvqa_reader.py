from ..utils.stream import ItemFeature
from .base_reader import IMIXDataReader
from imix.utils.common_function import update_d1_with_d2


class TextVQAReader(IMIXDataReader):

    def __init__(self, cfg):
        super().__init__(cfg)
        # assert self.default_feature, ('Not support non-default features now.')

    def __len__(self):
        return len(self.mix_annotations)

    def __getitem__(self, idx):
        annotation = self.mix_annotations[idx]
        feature = self.feature_obj[idx]
        global_feature, ocr_feature = {}, {}

        item_feature = ItemFeature(annotation)
        item_feature.error = False
        item_feature.tokens = annotation['question_tokens']
        item_feature.img_id = annotation['image_id']

        update_d1_with_d2(d1=item_feature, d2=feature)

        if self.global_feature_obj:
            global_feature = self.global_feature_obj[idx]
            global_feature.update({'features_global': global_feature.pop('features')})
            update_d1_with_d2(d1=item_feature, d2=global_feature)

        if self.ocr_feature_obj:
            ocr_feature = self.ocr_feature_obj[idx]
            ocr_feature.update({'features_ocr': ocr_feature.pop('features')})
            update_d1_with_d2(d1=item_feature, d2=ocr_feature)

        item_feature.error = None in [feature, global_feature, ocr_feature]

        return item_feature

    # def __getitem__(self, item):
    #     annotation = self.mix_annotations[item]
    #     # split = self.item_splits[item]
    #     item_feature = ItemFeature(annotation)
    #     item_feature.error = False
    #     '''
    #     for k, v in annotation.items():
    #         item_feature[k] = v
    #
    #     if split != 'test':
    #         item_feature.answers = annotation['answers']
    #     '''
    #     item_feature.tokens = annotation['question_tokens']
    #     item_feature.img_id = annotation['image_id']
    #
    #     feature_info = self.get_featureinfo_from_txns(self.feature_txns,
    #                                                   annotation['set_name'] + '/' + annotation['image_name'])
    #     for k, v in feature_info.items():
    #         item_feature[k] = item_feature.get(k, v)
    #         # item_feature[k] = v if item_feature.get(k) is None else item_feature[k]
    #
    #     feature_global_info = self.get_featureinfo_from_txns(self.feature_global_txns,
    #                                                          annotation['set_name'] + '/' + annotation['image_name'])
    #     feature_global_info['features_global'] = feature_global_info.pop('features')
    #
    #     for k, v in feature_global_info.items():
    #         item_feature[k] = item_feature.get(k, v)
    #         # item_feature[k] = v if item_feature.get(k) is None else item_feature[k]
    #
    #     feature_ocr_info = self.get_featureinfo_from_txns(self.feature_ocr_txns,
    #                                                       annotation['set_name'] + '/' + annotation['image_name'])
    #     feature_ocr_info['features_ocr'] = feature_ocr_info.pop('features')
    #     for k, v in feature_ocr_info.items():
    #         item_feature[k] = item_feature.get(k, v)
    #         # item_feature[k] = v if item_feature.get(k) is None else item_feature[k]
    #     item_feature.error = None in [feature_info, feature_global_info, feature_ocr_info]
    #
    #     return item_feature


#
#
#
# import yaml
# import numpy as np
# import os
# import io
# from tqdm import tqdm
#
# from .base_reader import BaseDataReader
# from ..utils.stream import ItemFeature
# from ..utils.phoc import phoc
#
#
# TEXTVQA_PATH_CONFIG = yaml.load(open("datasets/configs/dataset_textvqa.yaml"))["dataset_configs"]
#
# TEXTVQA_INFO_MAPPING = {
#     "split_idx": "split_idx",
#     "image_id": "img_id",
#     "image_width": "img_width",
#     "image_height": "img_height",
#     "question": "question",
#     "answers": "answers",
#     "valid_answers": "valid_answers"
# }
#
# class TextVQAReader(BaseDataReader):
#     """
#     Generate ItemFeatures:
#     img_id, img_width, img_height,
#     question, answers, valid_answers,
#     img_feats, img_boxes,
#     ocr_tokens, ocr_feats, ocr_boxes, ocr_fasttext, ocr_phoc
#     """
#     def __init__(self, splits:str, save_img_cache=True):
#         """
#         splits: in ("train", "val", "test")
#         save_img_cache: whether save image_featrues during loading batches
#         :param
#         """
#         super().__init__()
#         self.splits = splits.split(";")
#         self.ocr_dirs = [TEXTVQA_PATH_CONFIG["ocr_dir"][split][0] for split in self.splits]
#         self.img_dirs = [TEXTVQA_PATH_CONFIG["img_dir"][split][0] for split in self.splits]
#         self.frcnn_featrure_dirs = [TEXTVQA_PATH_CONFIG["frcnn_featrure"][split][0] for split in self.splits]
#         self.annotations_pathes = [TEXTVQA_PATH_CONFIG["annotations"][split][0] for split in self.splits]
#
#         self.fasttext_weights_path = TEXTVQA_PATH_CONFIG["fasttext_weights"][0]
#         self.load_fasttext_weights()
#         self.glove_weights_path = TEXTVQA_PATH_CONFIG["glove_weights"][0]
#         self.load_glove_weights()
#
#         self.feat_keys = ["img_feats", "img_boxes", "ocr_tokens", "ocr_feats",
#                           "ocr_boxes", "ocr_phoc", "ocr_fasttext"]
#         self.save_img_cache = save_img_cache
#         if self.save_img_cache:
#             self.cache = {}
#
#     def load_annotations(self):
#         self.annotations = []
#         for path in self.annotations_pathes:
#             infos = np.load(path, allow_pickle=True)
#             split_idx = self.annotations_pathes.index(path)
#             split = self.splits[split_idx]
#             [info.update({"split": split, "split_idx": split_idx}) for info in infos[1:]]
#             self.annotations.extend(infos[1:])
#
#     def get_length(self):
#         return len(self.annotations)
#
#     def __getitem__(self, item):
#         # item_feature = ItemFeature(self.annotations[item])
#         info_raw = self.annotations[item]
#         info = {}
#         for k, v in TEXTVQA_INFO_MAPPING.items():
#             info[v] = info_raw[k] if k in info_raw else None
#         img_id = info_raw["image_id"]
#         if self.save_img_cache and img_id in self.cache:
#             item_feature = self.cache[img_id].copy()
#         else:
#             item_feature = ItemFeature(info)
#             self.complete_info(item_feature)
#             if self.save_img_cache:
#                 self.cache[img_id] = item_feature
#         return item_feature
#
#     def complete_info(self, item_feature):
#         # load image features and bounding boxes
#
#         # load basic infos
#         img_id = item_feature["image_id"]
#         split_idx = item_feature["split_idx"]
#
#         # load image features and bounding boxes
#         img_feats = np.load(os.path.join(self.frcnn_featrure_dirs[split_idx], img_id+".npy"))
#         img_feats_info = np.load(os.path.join(self.frcnn_featrure_dirs[split_idx], img_id+"_info.npy"),
#         allow_pickle=True)[()]
#         img_boxes = img_feats_info["boxes"]
#         w = item_feature["img_width"]
#         h = item_feature["img_height"]
#         unif_st = np.array([w, h, w, h])
#         img_boxes /= unif_st
#
#         # load ocr tokens, features, bounding boxes and generate fasttext, glove, phoc features
#         ocr_feats = np.load(os.path.join(self.ocr_dirs[split_idx], img_id+".npy"))
#         ocr_feats_info = np.load(os.path.join(self.ocr_dirs[split_idx], img_id+"_info.npy"), allow_pickle=True)[()]
#         ocr_tokens = ocr_feats_info["ocr_tokens"]
#         ocr_boxes = ocr_feats_info["ocr_boxes"]
#         ocr_boxes /= unif_st
#         ocr_phoc_vectors = self.get_phoc(ocr_tokens)
#         ocr_fasttext_vectors = self.get_token_feat(ocr_tokens, "fasttext")
#         ocr_glove_vectors = self.get_token_feat(ocr_tokens, "glove")
#         ocr_order_vectors = np.eye(len(ocr_tokens), dtype=np.float32)
#
#         # save feats
#         item_feature["img_feats"] = img_feats
#         item_feature["img_boxes"] = img_boxes
#         item_feature["ocr_tokens"] = ocr_tokens
#         item_feature["ocr_feats"] = ocr_feats
#         item_feature["ocr_boxes"] = ocr_boxes
#         item_feature["ocr_phoc"] = ocr_phoc_vectors
#         item_feature["ocr_fasttext"] = ocr_fasttext_vectors
#         item_feature["ocr_glove"] = ocr_glove_vectors
#         item_feature["ocr_order_vectors"] = ocr_order_vectors
#
#     def get_phoc(self, tokens):
#         return np.array([phoc(token) for token in tokens])
#
#     def get_token_feat(self, tokens, switch):
#         """
#         :param tokens: ocr token list
#         :param switch: ("fasttext", "glove")
#         :return: featues
#         """
#         if switch not in ("fasttext", "glove"):
#             raise NotImplementedError
#         return np.array([self.get_single_token_feat(token, switch) for token in tokens])
#
#     def get_single_token_feat(self, token, switch):
#         if switch not in ("fasttext", "glove"):
#             raise NotImplementedError
#         words = [t for t in token.split(" ") if len(t) > 0]
#         if switch == "fasttext":
#             return np.mean(np.array([self.get_fasttext_single_word(word) for word in words]), 0)
#         elif switch == "glove":
#             return np.mean(np.array([self.get_glove_single_word(word) for word in words]), 0)
#
#     def load_fasttext_weights(self):
#         fin = io.open(self.fasttext_weights_path, "r", encoding="utf-8", newline="\n", errors="ignore")
#         # n, d = map(int, fin.readline().split())
#         self.fasttext_weights = {}
#         for line in fin:
#             tokens = line.rstrip().split(" ").copy()
#             vector = list(map(float, tokens[1:]))
#             token = tokens[0]
#             self.fasttext_weights[token] = vector
#
#     def load_glove_weights(self):
#         fin = io.open(self.glove_weights_path, "r", encoding="utf-8", newline="\n", errors="ignore")
#         self.glove_weights = {}
#         for line in fin.readline():
#             tokens = line.rstrip().split(" ").copy()
#             token = line[0]
#             vector = list(map(float, tokens[1:]))
#             self.glove_weights[token] = vector
#
#     def get_fasttext_single_word(self, word):
#         try:
#             return self.fasttext_weights[word]
#         except:
#             return ([0] * 300).copy()
#
#     def get_glove_single_word(self, word):
#         try:
#             return self.glove_weights[word]
#         except:
#             return ([0] * 300).copy()
