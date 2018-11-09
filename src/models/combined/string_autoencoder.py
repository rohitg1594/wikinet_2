# Autoencoder for string information of mentions and candidate strings
import torch.nn as nn
import torch
import torch.nn.functional as F


class StringAutoEncoder(nn.Module):

    def __init__(self, max_char_size=None, hidden_size=None, char_embs=None):
        super().__init__()

        self.max_char_size = max_char_size
        self.embs_size = char_embs.shape[1]

        self.char_embs = nn.Embedding(*char_embs.shape, padding_idx=0)
        self.char_embs.weight.data.copy_(torch.from_numpy(char_embs))

        self.lin1 = nn.Linear(self.max_char_size * self.embs_size, 2 * hidden_size)
        self.lin2 = nn.Linear(2 * hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 2 * hidden_size)
        self.lin4 = nn.Linear(2 * hidden_size, self.max_char_size * self.embs_size)

    def forward(self, input):
        b, max_char_size = input.shape

        input = self.char_embs(input).view(b, -1)
        hidden = self.lin2(F.relu(self.lin1(input)))
        output = self.lin4(F.relu(self.lin3(hidden)))

        return input, hidden, output
