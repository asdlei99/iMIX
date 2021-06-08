import unicodedata
import re
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


def _is_punctuation(char):
    if char == '-':
        return False
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


def _add_space_around_punc(string):
    pucs = [c for c in string if _is_punctuation(c)]
    for p in pucs:
        p_ = ' ' + p + ' '
        string = string.replace(p, p_)
    return string


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


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
        # if not os.path.isabs(vocab_file) and data_dir is not None:
        #     vocab_file = get_absolute_path(os.path.join(data_dir, vocab_file))

        # if not PathManager.exists(vocab_file):
        #     raise RuntimeError(f"Vocab file {vocab_file} for vocab dict doesn't exist")

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
