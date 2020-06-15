import numpy as np

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder

from ..data.sequence_labeling_data import get_vocab

ELMO = ELMoEmbedder("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-wiki_600k_steps.tar.gz")


def get_vectors():
    all_words = get_vocab()
    words_for_elmo = [[word] for word in all_words]
    return np.stack([np.zeros((1024,))] + ELMO(words_for_elmo))


class Model(nn.Module):

    def __init__(self, hidden_dim_1=512, hidden_dim_2=256, n_classes=3, keep_proba=0.4):

        super().__init__()

        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.vectors = get_vectors()
        self.vocab_size, self.embedding_dim = self.vectors.shape

        self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(self.vectors),
                                                            padding_idx=0)

        self.lstm_layer = nn.LSTM(self.embedding_dim, self.hidden_dim_1,
                                  batch_first=True, bidirectional=True, num_layers=2)

        self.linear_layer_1 = nn.Linear(self.hidden_dim_1 * 2, self.hidden_dim_1)
        self.linear_layer_2 = nn.Linear(self.hidden_dim_1, hidden_dim_2)
        self.linear_layer_3 = nn.Linear(hidden_dim_2, n_classes)

        self.dropout = nn.Dropout(p=keep_proba)

    def forward(self, x, sequence_length=32):
        x = self.embedding_layer(x)
        x_packed = pack_padded_sequence(x, [sequence_length] * len(x), batch_first=True)
        lstm_x, _ = self.lstm_layer(x_packed)
        lstm_output, _ = pad_packed_sequence(lstm_x, batch_first=True)
        lstm_output = self.dropout(lstm_output)

        linear1 = self.dropout(self.linear_layer_1(lstm_output))
        linear2 = self.dropout(self.linear_layer_2(linear1))
        out = self.linear_layer_3(linear2)

        return out