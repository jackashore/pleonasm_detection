import pandas as pd
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