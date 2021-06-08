import torch
from imix.utils.registry import Registry, build_from_cfg
import os
from imix.models.builder import EMBEDDING
from .baseprocessor import BaseProcessor
from imix.utils.third_party_libs import PathManager
import re

VOCAB = Registry('vocab')
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def tokenize(sentence, regex=SENTENCE_SPLIT_REGEX, keep=None, remove=None):
    if keep is None:
        keep = ["'s"]
    if remove is None:
        remove = [',', '?']
    sentence = sentence.lower()

    for token in keep:
        sentence = sentence.replace(token, ' ' + token)

    for token in remove:
        sentence = sentence.replace(token, '')

    tokens = regex.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def build_vocab(cfg):
    """Build vocab."""
    return build_from_cfg(cfg, VOCAB)


PREPROCESSOR = Registry('preprocessor')


def build_preprocessor(cfg):
    """Build preprocessor."""
    return build_from_cfg(cfg, PREPROCESSOR)


def load_str_list(fname):
    with PathManager.open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


@EMBEDDING.register_module()
class SimpleWordProcessor(BaseProcessor):
    """Tokenizes a word and processes it.

    Attributes:
        tokenizer (function): Type of tokenizer to be used.
    """

    def __init__(self, *args, **kwargs):
        from imix.utils.third_party_libs import word_tokenize

        self.tokenizer = word_tokenize

    def __call__(self, item, *args, **kwargs):
        return {'text': self.tokenizer(item['text'], *args, **kwargs)}


@EMBEDDING.register_module()
class VocabDict:
    UNK_TOKEN = '<unk>'
    PAD_TOKEN = '<pad>'
    START_TOKEN = '<s>'
    END_TOKEN = '</s>'

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, vocab_file, data_dir=None):
        if not os.path.isabs(vocab_file) and data_dir is not None:
            vocab_file = os.path.abspath(os.path.join(data_dir, vocab_file))

        if not PathManager.exists(vocab_file):
            raise RuntimeError(f"Vocab file {vocab_file} for vocab dict doesn't exist")

        self.word_list = load_str_list(vocab_file)
        self._build()

    def _build(self):
        if self.UNK_TOKEN not in self.word_list:
            self.word_list = [self.UNK_TOKEN] + self.word_list

        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}

        # String (word) to integer (index) dict mapping
        self.stoi = self.word2idx_dict
        # Integer to string (word) reverse mapping
        self.itos = self.word_list
        self.num_vocab = len(self.word_list)

        self.UNK_INDEX = (self.word2idx_dict[self.UNK_TOKEN] if self.UNK_TOKEN in self.word2idx_dict else None)

        self.PAD_INDEX = (self.word2idx_dict[self.PAD_TOKEN] if self.PAD_TOKEN in self.word2idx_dict else None)

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def __len__(self):
        return len(self.word_list)

    def get_size(self):
        return len(self.word_list)

    def get_unk_index(self):
        return self.UNK_INDEX

    def get_unk_token(self):
        return self.UNK_TOKEN

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_INDEX is not None:
            return self.UNK_INDEX
        else:
            raise ValueError('word %s not in dictionary \
                             (while dictionary does not contain <unk>)' % w)

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]
        return inds


@EMBEDDING.register_module()
class VQAAnswerProcessor(BaseProcessor):
    """Processor for generating answer scores for answers passed using VQA
    accuracy formula. Using VocabDict class to represent answer vocabulary, so
    parameters must specify "vocab_file". "num_answers" in parameter config
    specify the max number of answers possible. Takes in dict containing
    "answers" or "answers_tokens". "answers" are preprocessed to generate
    "answers_tokens" if passed.

    Args:
        config (DictConfig): Configuration for the processor

    Attributes:
        answer_vocab (VocabDict): Class representing answer vocabulary
    """

    DEFAULT_NUM_ANSWERS = 10

    def __init__(self, answer_vocab, preprocessor, num_answers, *args, **kwargs):
        # self.writer = registry.get("writer")
        # if not hasattr(config, "vocab_file"):
        #     raise AttributeError(
        #         "'vocab_file' argument required, but not "
        #         "present in AnswerProcessor's config"
        #     )

        # self.answer_vocab = VocabDict(config.vocab_file, *args, **kwargs)
        self.answer_vocab = build_vocab(answer_vocab)

        self.preprocessor = build_preprocessor(preprocessor)

        self.preprocessor = None

        # if hasattr(config, "preprocessor"):
        #     self.preprocessor = Processor(config.preprocessor)
        #
        #     if self.preprocessor is None:
        #         raise ValueError(
        #             f"No processor named {config.preprocessor} is defined."
        #         )
        self.num_answers = num_answers
        # if hasattr(config, "num_answers"):
        #     self.num_answers = config.num_answers
        # else:
        #     self.num_answers = self.DEFAULT_NUM_ANSWERS
        #     warnings.warn(
        #         "'num_answers' not defined in the config. "
        #         "Setting to default of {}".format(self.DEFAULT_NUM_ANSWERS)
        #     )

    def __call__(self, item):
        """Takes in dict with answers or answers_tokens, and returns back a
        dict with answers (processed), "answers_indices" which point to indices
        of the answers if present and "answers_scores" which represent VQA
        style scores for the answers.

        Args:
            item (Dict): Dict containing answers or answers_tokens

        Returns:
            Dict: Processed answers, indices and scores.
        """
        tokens = None

        if not isinstance(item, dict):
            raise TypeError("'item' passed to processor must be a dict")

        if 'answer_tokens' in item:
            tokens = item['answer_tokens']
        elif 'answers' in item:
            if self.preprocessor is None:
                raise AssertionError("'preprocessor' must be defined if you " "don't pass 'answer_tokens'")

            tokens = [self.preprocessor({'text': answer})['text'] for answer in item['answers']]
        else:
            raise AssertionError("'answers' or 'answer_tokens' must be passed" ' to answer processor in a dict')

        tokens = self._increase_to_ten(tokens)
        answers_indices = torch.zeros(self.DEFAULT_NUM_ANSWERS, dtype=torch.long)
        answers_indices.fill_(self.answer_vocab.get_unk_index())

        for idx, token in enumerate(tokens):
            answers_indices[idx] = self.answer_vocab.word2idx(token)

        answers_scores = self.compute_answers_scores(answers_indices)

        return {
            'answers': tokens,
            'answers_indices': answers_indices,
            'answers_scores': answers_scores,
        }

    def get_vocab_size(self):
        """Get vocab size of the answer vocabulary. Can also include soft copy
        dynamic answer space size.

        Returns:
            int: size of the answer vocabulary
        """
        return self.answer_vocab.num_vocab

    def get_true_vocab_size(self):
        """True vocab size can be different from normal vocab size in some
        cases such as soft copy where dynamic answer space is added.

        Returns:
            int: True vocab size.
        """
        return self.answer_vocab.num_vocab

    def word2idx(self, word):
        """Convert a word to its index according to vocabulary.

        Args:
            word (str): Word to be converted to index.

        Returns:
            int: Index of the word.
        """
        return self.answer_vocab.word2idx(word)

    def idx2word(self, idx):
        """Index to word according to the vocabulary.

        Args:
            idx (int): Index to be converted to the word.

        Returns:
            str: Word corresponding to the index.
        """
        return self.answer_vocab.idx2word(idx)

    def compute_answers_scores(self, answers_indices):
        """Generate VQA based answer scores for answers_indices.

        Args:
            answers_indices (torch.LongTensor): tensor containing indices of the answers

        Returns:
            torch.FloatTensor: tensor containing scores.
        """
        scores = torch.zeros(self.get_vocab_size(), dtype=torch.float)
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

            if answer != self.answer_vocab.UNK_INDEX:
                scores[answer] = avg_acc

        return scores

    def _increase_to_ten(self, tokens):
        while len(tokens) < self.DEFAULT_NUM_ANSWERS:
            tokens += tokens[:self.DEFAULT_NUM_ANSWERS - len(tokens)]

        return tokens
