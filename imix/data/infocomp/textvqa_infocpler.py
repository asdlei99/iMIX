from collections import defaultdict

import numpy as np
import torch

from ..utils.common import VocabDict
from ..utils.stream import ItemFeature
from .base_infocpler import BaseInfoCpler


class TextVQAInfoCpler(BaseInfoCpler):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.answer_vocab = VocabDict(cfg.mix_vocab.fixed_answer_vocab_textvqa_5k)
        self.PAD_IDX = self.answer_vocab.word2idx('<pad>')
        self.BOS_IDX = self.answer_vocab.word2idx('<s>')
        self.EOS_IDX = self.answer_vocab.word2idx('</s>')
        self.UNK_IDX = self.answer_vocab.UNK_INDEX
        # make sure PAD_IDX, BOS_IDX and PAD_IDX are valid (not <unk>)
        assert self.PAD_IDX != self.answer_vocab.UNK_INDEX
        assert self.BOS_IDX != self.answer_vocab.UNK_INDEX
        assert self.EOS_IDX != self.answer_vocab.UNK_INDEX
        assert self.PAD_IDX == 0
        self.num_answers = 10
        self.max_length = cfg.max_txt_lenth
        self.max_copy_steps = cfg.max_copy_steps
        self.match_answer_to_unk = False

    def get_vocab_size(self):
        answer_vocab_nums = self.answer_vocab.num_vocab
        answer_vocab_nums += self.max_length
        return answer_vocab_nums

    def completeInfo(self, item_feature: ItemFeature):
        if self.if_bert:
            tokens = self.tokenizer.tokenize(item_feature.question.strip())

            tokens = self.tokenizer.get_limited_tokens(tokens, self.max_seq_length - 2)
            tokens, input_lm_label_ids = self.tokenizer.random_mask_tokens(tokens, self.word_mask_ratio)
            tokens = [self._CLS_TOKEN] + tokens + [self._SEP_TOEKN]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(tokens)
            input_segment = [0] * len(tokens)
            input_lm_label_ids = [-1] * len(tokens)
            to_extd_length = self.max_seq_length - len(input_ids)
            self.info_extend(to_extd_length, (input_ids, int(self.pad_idx)), (input_mask, 0), (input_segment, 0),
                             (input_lm_label_ids, -1))
            # while len(input_ids) < self.max_seq_length:
            #    input_ids.append(int(self.pad_idx))
            #    input_mask.append(0)
            #    input_segment.append(0)
            #    input_lm_label_ids.append(-1)
            item_feature.input_segment = torch.tensor(input_segment, dtype=torch.int)
            item_feature.input_lm_label_ids = torch.tensor(input_lm_label_ids, dtype=torch.long)

        else:
            tokens = item_feature.tokens
            if len(tokens) > self.max_seq_length:
                tokens = tokens[:self.max_seq_length]
            input_ids = [self.stoi[t] for t in tokens]
            input_mask = [1] * len(tokens)
            to_extd_length = self.max_seq_length - len(input_ids)
            self.info_extend(to_extd_length, (input_ids, int(self.pad_idx)), (input_mask, 0))
            # while len(input_ids) < self.max_seq_length:
            #    input_ids.append(int(self.pad_idx))
            #    input_mask.append(0)

        # ocr vectors

        ocr_tokens = self.tokenizer.get_limited_tokens(item_feature.ocr_tokens, self.max_ocr_length)
        ocr_tokens = [self.word_tokenize(tmp) for tmp in ocr_tokens]
        ocr_tokens, ocr_length = self._pad_tokens(ocr_tokens)
        item_feature.ocr_vectors_glove = self.get_tokens_glove_vectors(ocr_tokens)
        item_feature.ocr_vectors_order = self.get_tokens_order_vectors(ocr_tokens)
        item_feature.ocr_vectors_phoc = self.get_tokens_phoc_vectors(ocr_tokens)
        item_feature.ocr_vectors_fasttext = self.get_tokens_fasttext_vectors(ocr_tokens)

        # ocr features and bboxes
        features_ocr = torch.zeros(
            (self.max_ocr_length,
             item_feature.features_ocr.shape[1] if item_feature.features_ocr is not None else 2048),
            dtype=torch.float)
        bbox_ocr_normalized = torch.zeros(
            (self.max_ocr_length,
             item_feature.ocr_normalized_boxes.shape[1] if item_feature.ocr_normalized_boxes is not None else 4),
            dtype=torch.float)
        if item_feature.features_ocr is not None:
            limit = min(self.max_ocr_length, len(item_feature.features_ocr))
            features_ocr[:limit] = torch.tensor(item_feature.features_ocr[:limit])
            bbox_ocr_normalized[:limit] = torch.tensor(item_feature.ocr_normalized_boxes[:limit])
        item_feature.features_ocr = features_ocr
        item_feature.ocr_normalized_boxes = bbox_ocr_normalized

        # features and bboxes
        img_h = item_feature.image_height
        img_w = item_feature.image_width
        item_feature.bbox = self._get_bbox_from_normalized(item_feature.obj_normalized_boxes, img_h, img_w)
        item_feature.bbox_normalized = item_feature.obj_normalized_boxes
        item_feature.bbox_ocr = self._get_bbox_from_normalized(item_feature.ocr_normalized_boxes, img_h, img_w)
        item_feature.bbox_ocr_normalized = item_feature.ocr_normalized_boxes

        item_feature.input_ids = torch.tensor(input_ids, dtype=torch.long)
        item_feature.input_mask = torch.tensor(input_mask, dtype=torch.int)

        item_feature.qa_ids = [self.qa_ans2id[ans] for ans in item_feature.answers if ans in self.qa_ans2id]
        # item_feature.qa_allids = [self.qa_ans2id[ans] for ans in item_feature.all_answers if ans in self.qa_ans2id]
        item_feature.answers_scores = self.compute_answers_scores(item_feature.answers)
        answers = self.compute_special_answers(item_feature, ocr_tokens)
        for k, v in answers.items():
            item_feature[k] = v
        return item_feature

    def get_tokens_order_vectors(self, tokens):
        vector = torch.full((self.max_ocr_length, self.max_length), fill_value=self.PAD_INDEX, dtype=torch.float)
        vector[:len(tokens), :len(tokens)] = torch.from_numpy(np.eye(len(tokens), dtype=np.float32))
        return vector

    def compute_answers_scores(self, answers):
        gt_answers = list(enumerate(answers))
        unique_answers = sorted(set(answers))
        unique_answer_scores = [0] * len(unique_answers)
        for idx, unique_answer in enumerate(unique_answers):
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [item for item in other_answers if item[1] == unique_answer]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[idx] = sum(accs) / len(accs)
        unique_answer2score = {a: s for a, s in zip(unique_answers, unique_answer_scores)}
        return unique_answer2score

    def match_answer_to_vocab_ocr_seq(self, answer, vocab2idx_dict, ocr2inds_dict, max_match_num=20):
        """Match an answer to a list of sequences of indices each index
        corresponds to either a fixed vocabulary or an OCR token (in the index
        address space, the OCR tokens are after the fixed vocab)"""
        num_vocab = len(vocab2idx_dict)

        answer_words = answer.split()
        answer_word_matches = []
        for word in answer_words:
            # match answer word to fixed vocabulary
            matched_inds = []
            if word in vocab2idx_dict:
                matched_inds.append(vocab2idx_dict.get(word))
            # match answer word to OCR
            # we put OCR after the fixed vocabulary in the answer index space
            # so add num_vocab offset to the OCR index
            matched_inds.extend([num_vocab + idx for idx in ocr2inds_dict[word]])
            if len(matched_inds) == 0:
                if self.match_answer_to_unk:
                    matched_inds.append(vocab2idx_dict.get('<unk>'))
                else:
                    return []
            answer_word_matches.append(matched_inds)

        # expand per-word matched indices into the list of matched sequences
        if len(answer_word_matches) == 0:
            return []
        idx_seq_list = [()]
        for matched_inds in answer_word_matches:
            idx_seq_list = [seq + (idx, ) for seq in idx_seq_list for idx in matched_inds]
            if len(idx_seq_list) > max_match_num:
                idx_seq_list = idx_seq_list[:max_match_num]

        return idx_seq_list

    def _pad_tokens(self, tokens):
        padded_tokens = [self.PAD_TOKEN] * self.max_length
        token_length = min(len(tokens), self.max_length)
        padded_tokens[:token_length] = tokens[:token_length]
        token_length = torch.tensor(token_length, dtype=torch.long)
        return padded_tokens, token_length

    def compute_special_answers(self, item_feature, ocr_tokens):
        unique_answer2score = item_feature.answers_scores
        scores = torch.zeros(self.max_copy_steps, self.get_vocab_size(), dtype=torch.float)
        answers = item_feature.answers
        ocr2inds_dict = defaultdict(list)
        for idx, token in enumerate(ocr_tokens):
            ocr2inds_dict[token].append(idx)

        answer_dec_inds = []
        for a in answers:
            tmp = self.match_answer_to_vocab_ocr_seq(a, self.answer_vocab.word2idx_dict, ocr2inds_dict)
            answer_dec_inds.append(tmp)

        all_idx_seq_list = []
        for answer, idx_seq_list in zip(answers, answer_dec_inds):
            all_idx_seq_list.extend(idx_seq_list)
            # fill in the soft score for the first decoding step
            score = unique_answer2score[answer]
            for idx_seq in idx_seq_list:
                score_idx = idx_seq[0]
                # the scores for the decoding Step 0 will be the maximum
                # among all answers starting with that vocab
                # for example:
                # if "red apple" has score 0.7 and "red flag" has score 0.8
                # the score for "red" at Step 0 will be max(0.7, 0.8) = 0.8
                scores[0, score_idx] = max(scores[0, score_idx], score)

        # train_prev_inds is the previous prediction indices in auto-regressive
        # decoding
        train_prev_inds = torch.zeros(self.max_copy_steps, dtype=torch.long)
        # train_loss_mask records the decoding steps where losses are applied
        train_loss_mask = torch.zeros(self.max_copy_steps, dtype=torch.float)
        if len(all_idx_seq_list) > 0:
            # sample a random decoding answer sequence for teacher-forcing
            idx_seq = all_idx_seq_list[np.random.choice(len(all_idx_seq_list))]

            dec_step_num = min(1 + len(idx_seq), self.max_copy_steps)
            train_loss_mask[:dec_step_num] = 1.0

            train_prev_inds[0] = self.BOS_IDX
            for t in range(1, dec_step_num):
                train_prev_inds[t] = idx_seq[t - 1]
                score_idx = idx_seq[t] if t < len(idx_seq) else self.EOS_IDX
                scores[t, score_idx] = 1.0
        else:
            idx_seq = ()

        answer_info = {
            'answers': answers,
            'answers_scores': scores,
            'sampled_idx_seq': idx_seq,
            'train_prev_inds': train_prev_inds,
            'train_loss_mask': train_loss_mask,
        }

        return answer_info
