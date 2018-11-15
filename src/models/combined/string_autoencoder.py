# Autoencoder for string information of mentions and candidate strings
import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np


class StringAutoEncoder(nn.Module):

    def __init__(self, max_char_size=None, hidden_size=None, char_embs=None, dp=None, activate=None, norm=False):
        super().__init__()

        self.max_char_size = max_char_size
        self.embs_size = char_embs.shape[1]
        self.norm = norm

        self.char_embs = nn.Embedding(*char_embs.shape, padding_idx=0)
        if isinstance(char_embs, np.ndarray):
            char_embs = torch.from_numpy(char_embs)
        self.char_embs.weight.data.copy_(char_embs)

        self.lin1 = nn.Linear(self.max_char_size * self.embs_size, 2 * hidden_size)
        self.lin2 = nn.Linear(2 * hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 2 * hidden_size)
        self.lin4 = nn.Linear(2 * hidden_size, self.max_char_size * self.embs_size)

        if activate == 'relu':
            self.activate = F.relu
        elif activate == 'tanh':
            self.activate = F.tanh
        elif activate == 'sigmoid':
            self.activate = F.sigmoid
        else:
            self.activate = None

        self.dp = nn.Dropout(dp)

    def forward(self, input):
        input = self.char_embs(input).view(*input.shape[:-1], -1)
        input = self.dp(input)
        hidden = self.lin2(F.relu(self.dp(self.lin1(input))))
        if self.norm:
            hidden = F.normalize(hidden)
        if self.activate:
            hidden = self.activate(hidden)
        output = self.lin4(F.relu(self.lin3(hidden)))

        return input, hidden, output
