# This is the yamada MLP for entity linking
import torch.nn as nn
import torch


class YamadaBase(nn.Module):
    def __init__(self, yamada_model=None, args=None):
        super().__init__()

        word_embs = yamada_model['word_emb']
        ent_embs = yamada_model['ent_emb']
        W = yamada_model['W']
        b = yamada_model['b']

        self.emb_dim = word_embs.shape[1]
        self.num_ent = ent_embs.shape[0]
        self.args = args

        # Words
        self.word_embs = nn.Embedding(*word_embs.shape, padding_idx=0, sparse=True)
        self.word_embs.weight.data.copy_(torch.from_numpy(word_embs))
        self.word_embs.weight.requires_grad = False

        # Entities
        self.ent_embs = nn.Embedding(*ent_embs.shape, padding_idx=0, sparse=True)
        self.ent_embs.weight.data.copy_(torch.from_numpy(ent_embs))
        self.ent_embs.weight.requires_grad = False

        # Pre trained linear layer
        self.orig_linear = nn.Linear(word_embs.shape[1], ent_embs.shape[1])
        self.orig_linear.weight.data.copy_(torch.from_numpy(W))
        self.orig_linear.bias.data.copy_(torch.from_numpy(b))
        self.orig_linear.weight.requires_grad = True

        self.dropout = nn.Dropout(self.args.dp)

    def forward(self, inputs):

        raise NotImplementedError
