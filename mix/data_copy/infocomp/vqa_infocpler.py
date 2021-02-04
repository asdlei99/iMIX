import torch
from collections import defaultdict
from ..vqadata.stream import ItemFeature
import logging

WORD_MASK_RATIO = 0.15
MAX_SEQ_LENGTH = 14


class VQAInfoCpler(object):

  PAD_TOKEN = '<pad>'
  SOS_TOKEN = '<s>'
  EOS_TOKEN = '</s>'
  UNK_TOKEN = '<unk>'

  PAD_INDEX = 0
  SOS_INDEX = 1
  EOS_INDEX = 2
  UNK_INDEX = 3

  def __init__(self, cfg):
    # self.tokenizer = BertTokenizer.from_pretrained(
    #     "bert-base-uncased",
    #     do_lower_case=True
    # )
    # logger = logging.getLogger(__name__)

    self.max_seq_length = MAX_SEQ_LENGTH
    # self.vocab_path = vqa_path_config["mmf_vocab"]["vocabulart_100k"]

    # self.glove_weights_path = vqa_path_config["glove_weights"]
    # self.answer_vocab_path = vqa_path_config["mmf_vocab"]["answers_vqa"]

    self.glove_weights_path = cfg.glove_weights
    self.answer_vocab_path = cfg.mmf_vocab.answers_vqa

    self.load_glove_weights()
    with open(self.answer_vocab_path) as f:
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
    # vocab_file = vqa_path_config["mmf_vocab"]["vocabulart_100k"]
    vocab_file = cfg.mmf_vocab.vocabulart_100k
    if vocab_file is not None:
      # if not os.path.isabs(vocab_file) and data_dir is not None:
      #     vocab_file = os.path.join(data_dir, vocab_file)
      #     vocab_file = get_absolute_path(vocab_file)

      # if not PathManager.exists(vocab_file):
      #     raise RuntimeError("Vocab not found at " + vocab_file)

      with open(vocab_file, 'r') as f:
        for line in f:
          self.itos[index] = line.strip()
          self.word_dict[line.strip()] = index
          index += 1

    self.word_dict[self.SOS_TOKEN] = self.SOS_INDEX
    self.word_dict[self.EOS_TOKEN] = self.EOS_INDEX
    self.word_dict[self.PAD_TOKEN] = self.PAD_INDEX
    self.word_dict[self.UNK_TOKEN] = self.UNK_INDEX
    # Return unk index by default
    self.stoi = defaultdict(self.get_unk_index)
    self.stoi.update(self.word_dict)
    #print('xiix')
    # logger.info("VQAInfoCpler success")

  def get_unk_index(self):
    return self.UNK_INDEX

  def completeInfo(self, itemFeature: ItemFeature):
    tokens = itemFeature.tokens
    # tokens = self.tokenizer.tokenize(question.strip())

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
    self.vocabs = glove[0]
    self.vocab_dict = glove[1]
    self.glove_weights = glove[2]

  def get_glove_single_word(self, word):
    try:
      return self.glove_weights[self.vocab_dict[word]]
    except:
      return ([0] * 300).copy()

  def get_glove_single_id(self, id):
    try:
      return self.glove_weights[id]
    except:
      return torch.zeros((300,))
