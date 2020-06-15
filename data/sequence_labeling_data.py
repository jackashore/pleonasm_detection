import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..data.preprocessing import tokenize

TRAIN = pd.read_excel('train.xlsx')
TEST = pd.read_excel('test.xlsx')
DATA = pd.concat([TRAIN, TEST])

def get_vocab():
    all_words = []

    for text in tqdm(DATA.text):
        all_words.extend(tokenize(text))

    return list(set(all_words))


def get_word2index():
    all_words = get_vocab()
    word2index = {w: i for i, w in enumerate(all_words, 1)}
    word2index['PAD'] = 0

    return word2index


class ELMoData(Dataset):

    def __init__(self, x_data, sequence_length=32, pad_token='PAD', verbose=True):

        super().__init__()

        self.x_data = []
        self.y_data = []

        self.sequence_length = sequence_length
        self.word2index = get_word2index()
        self.pad_token = pad_token
        self.tag2index = {'O': 1, 'B': 2, self.pad_token: 0}

        self.load(x_data, verbose=verbose)

    def process_targets(self, targets):
        return [self.tag2index[tag] for tag in targets]

    def load(self, data, verbose=True):
        data_iterator = tqdm(data, desc='Loading data', disable=not verbose)

        for pair in data_iterator:
            text = pair[0]
            tags = pair[1].split()

            tokens = tokenize(text)
            tokens = self.indexing(tokens)
            tokens = self.padding(tokens)
            self.x_data.append(tokens)

            targets = self.process_targets(tags)
            targets = self.padding(targets)
            self.y_data.append(targets)

    def padding(self, sequence):
        if len(sequence) > self.sequence_length:
            sequence = sequence[:self.sequence_length]
        elif len(sequence) < self.sequence_length:
            sequence = sequence + [self.tag2index[self.pad_token]] * (self.sequence_length - len(sequence))
        return sequence

    def indexing(self, tokenized_text):
        return [self.word2index[token] for token in tokenized_text if token in self.word2index]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):

        x = self.x_data[idx]
        x = torch.Tensor(x).long()

        y = self.y_data[idx]
        y = torch.Tensor(y).long()

        return x, y
