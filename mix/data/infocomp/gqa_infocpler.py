import torch
from collections import defaultdict
# from ..vqadata.stream import ItemFeature
import logging

from ..utils.tokenization import BertTokenizer
from ..utils.stream import ItemFeature

WORD_MASK_RATIO = 0.15
MAX_SEQ_LENGTH = 14


class GQAInfoCpler(object):

  def __init__(self, cfg):
    # logger = logging.getLogger(__name__)

    self.if_bert = cfg.if_bert
    self._init_tokens()

    self.max_seq_length = cfg.get('max_seg_lenth', 14)
    self.word_mask_ratio = cfg.get('word_mask_ratio', 0.15)

    self.vocab_name = cfg.get('vocab_name',
                              'vocabulary_gqa')  ### bert for vocabulart_100k
    self.vocab_path = cfg['mix_vocab'][self.vocab_name]

    self.vocab_answer_name = cfg.get('vocab_answer_name', 'answers_gqa')
    self.vocab_answer_path = cfg['mix_vocab'][self.vocab_answer_name]

    self.glove_name = cfg.get('glove_name', 'glove6b300d')
    self.glove_weights_path = cfg['glove_weights'][self.glove_name]

    self.load_glove_weights()
    self.load_vocab()

    #print('xiix')
    # logger.info("VQAInfoCpler success")

  def completeInfo(self, itemFeature: ItemFeature):
    if self.if_bert:
      return self.completeBertInfo(itemFeature)
    else:
      return self.completeNormalInfo(itemFeature)

  def completeNormalInfo(self, itemFeature):
    tokens = itemFeature.tokens
    itemFeature = self.compute_bboxInfo(itemFeature)
    if len(tokens) > self.max_seq_length:
      tokens = tokens[:self.max_seq_length]

    input_ids = [self.stoi[t] for t in tokens]
    input_mask = [1] * len(input_ids)
    while len(input_ids) < self.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
    itemFeature.input_ids = torch.tensor(input_ids, dtype=torch.long)
    itemFeature.input_mask = torch.tensor(input_mask, dtype=torch.bool)
    # itemFeature.feature_question = torch.stack(list(map(self.get_glove_single_id, input_ids)))

    if itemFeature.answers is not None:
      itemFeature.answers = self._increase_to_ten(itemFeature.answers)
      itemFeature.qa_ids = [
          self.qa_ans2id[ans]
          for ans in itemFeature.answers
          if ans in self.qa_ans2id
      ]
      itemFeature.qa_allids = [
          self.qa_ans2id[ans]
          for ans in itemFeature.all_answers
          if ans in self.qa_ans2id
      ]
      itemFeature.answers_scores = self.compute_answers_scores(
          torch.Tensor(itemFeature.qa_ids))
    return itemFeature

  def compute_bboxInfo(self, itemFeature):
    bbox = itemFeature['bbox']
    image_w = itemFeature['image_width']
    image_h = itemFeature['image_height']
    image_location = torch.zeros((bbox.shape[0], 4), dtype=torch.float)
    image_location[:, :4] = torch.from_numpy(bbox)
    # image_location[:, 4] = (
    #     (image_location[:, 3] - image_location[:, 1])
    #     * (image_location[:, 2] - image_location[:, 0])
    #     / (image_w * image_h)
    # )
    image_location[:, 0] = image_location[:, 0] / image_w
    image_location[:, 1] = image_location[:, 1] / image_h
    image_location[:, 2] = image_location[:, 2] / image_w
    image_location[:, 3] = image_location[:, 3] / image_h
    itemFeature['bbox'] = image_location
    return itemFeature

  def completeBertInfo(self, itemFeature):
    tokens = self.tokenizer.tokenize(itemFeature.question_str.strip())
    tokens = self.tokenizer.get_limited_tokens(tokens, self.max_seq_length - 2)
    tokens, input_lm_label_ids = self.tokenizer.random_mask_tokens(
        tokens, self.word_mask_ratio)
    tokens = [self._CLS_TOKEN] + tokens + [self._SEP_TOEKN]

    itemFeature = self.compute_bboxInfo(itemFeature)

    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] + [1] * (len(tokens) - 2) + [1]
    input_segment = [0] + [0] * (len(tokens) - 2) + [0]
    input_lm_label_ids = [-1] + [-1] * (len(tokens) - 2) + [-1]
    while len(input_ids) < self.max_seq_length:
      input_ids.append(int(self.pad_idx))
      input_mask.append(0)
      input_segment.append(0)
      input_lm_label_ids.append(-1)

    itemFeature.input_ids = torch.tensor(
        input_ids, dtype=torch.long)  # token ids
    itemFeature.input_mask = torch.tensor(
        input_mask, dtype=torch.long)  # token mask
    itemFeature.input_segment = torch.tensor(
        input_segment, dtype=torch.long)  # token segments
    itemFeature.input_lm_label_ids = torch.tensor(
        input_lm_label_ids, dtype=torch.long)  # token mlm labels
    itemFeature.qa_ids = [
        self.qa_ans2id[ans]
        for ans in itemFeature.answers
        if ans in self.qa_ans2id
    ]
    itemFeature.qa_allids = [
        self.qa_ans2id[ans]
        for ans in itemFeature.all_answers
        if ans in self.qa_ans2id
    ]
    itemFeature.answers_scores = self.compute_answers_scores(
        torch.Tensor(itemFeature.qa_ids))

    return itemFeature

  def compute_answers_scores(self, answers_indices):
    """Generate VQA based answer scores for answers_indices.

    Args:
        answers_indices (torch.LongTensor): tensor containing indices of the answers

    Returns:
        torch.FloatTensor: tensor containing scores.
    """
    scores = torch.zeros(len(self.qa_ans2id), dtype=torch.float)
    gt_answers = list(enumerate(answers_indices))
    unique_answers = set(answers_indices.tolist())

    for answer in unique_answers:
      accs = []
      for gt_answer in gt_answers:
        other_answers = [item for item in gt_answers if item != gt_answer]
        matching_answers = [item for item in other_answers if item[1] == answer]
        acc = min(1, float(len(matching_answers)) / 3)
        accs.append(acc)
      avg_acc = sum(accs) / len(accs)
      if answer != 0:
        scores[int(answer)] = avg_acc
    return scores

  def _increase_to_ten(self, tokens):
    while len(tokens) < self.DEFAULT_NUM_ANSWERS:
      tokens += tokens[:self.DEFAULT_NUM_ANSWERS - len(tokens)]
    return tokens

  def load_glove_weights(self):
    glove = torch.load(self.glove_weights_path)
    self.glove_vocabs = glove[0]
    self.glove_vocab_dict = glove[1]
    self.glove_weights = glove[2]

  def get_glove_single_word(self, word):
    try:
      return self.glove_weights[self.glove_vocab_dict[word]]
    except:
      return ([0] * 300).copy()

  def get_glove_single_id(self, id):
    if id == self.pad_idx:
      return torch.zeros((300,))
    try:
      return self.glove_weights[id]
    except:
      return torch.zeros((300,))

  def load_vocab(self):
    with open(self.vocab_answer_path) as f:
      raw_qa_vocab = f.readlines()
    self.qa_id2ans = [t.strip() for t in raw_qa_vocab]
    self.qa_ans2id = {k: v for v, k in enumerate(self.qa_id2ans)}
    self.DEFAULT_NUM_ANSWERS = 10

    self.word_dict = {}
    self.itos = {}

    self.itos[self.PAD_INDEX] = self.PAD_TOKEN
    self.itos[self.SOS_INDEX] = self.SOS_TOKEN
    self.itos[self.EOS_INDEX] = self.EOS_TOKEN
    self.itos[self.UNK_INDEX] = self.UNK_TOKEN
    self.word_dict[self.SOS_TOKEN] = self.SOS_INDEX
    self.word_dict[self.EOS_TOKEN] = self.EOS_INDEX
    self.word_dict[self.PAD_TOKEN] = self.PAD_INDEX
    self.word_dict[self.UNK_TOKEN] = self.UNK_INDEX

    index = len(self.itos.keys())
    self.total_predefined = len(self.itos.keys())
    with open(self.vocab_path, 'r') as f:
      for line in f:
        self.itos[index] = line.strip()
        self.word_dict[line.strip()] = index
        index += 1

    self.stoi = defaultdict(self.get_unk_index)
    self.stoi.update(self.word_dict)

  def get_unk_index(self):
    return self.UNK_INDEX

  def _init_tokens(self):
    self.tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)
    self.PAD_TOKEN = '<pad>'
    self.SOS_TOKEN = '<s>'
    self.EOS_TOKEN = '</s>'
    self.UNK_TOKEN = '<unk>'
    self.PAD_INDEX = 0
    self.SOS_INDEX = 1
    self.EOS_INDEX = 2
    self.UNK_INDEX = 3
    self._MASK_TOKEN = '[MASK]'
    self._SEP_TOEKN = '[SEP]'
    self._CLS_TOKEN = '[CLS]'
    self._PAD_TOKEN = '[PAD]'
    self.pad_idx = self.tokenizer.vocab[self._PAD_TOKEN]
