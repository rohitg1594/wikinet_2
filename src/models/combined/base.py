# Combined models are based on this model
import torch
import torch.nn as nn


class CombinedBase(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        # Unpack args
        args = kwargs['args']
        word_embs = kwargs['word_embs']
        ent_embs = kwargs['ent_embs']
        W = kwargs['W']
        b = kwargs['b']
        gram_embs = kwargs['gram_embs']

        self.args = args

        # Word embeddings
        self.word_embs = nn.Embedding(*word_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.word_embs.weight.data.copy_(torch.from_numpy(word_embs))
        self.word_embs.weight.requires_grad = self.args.train_word

        # Gram embeddings
        self.gram_embs = nn.Embedding(*gram_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.gram_embs.weight.data.copy_(torch.from_numpy(gram_embs))
        self.gram_embs.weight.requires_grad = self.args.train_gram

        # Entity embeddings
        self.ent_embs = nn.Embedding(*ent_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.ent_embs.weight.data.copy_(torch.from_numpy(ent_embs))
        self.ent_embs.weight.requires_grad = self.args.train_ent

        # Linear Layer
        self.orig_linear = nn.Linear(*W.shape)
        if self.args.init_embs == 'yamada':
            self.orig_linear.weight.data.copy_(torch.from_numpy(W))
            self.orig_linear.bias.data.copy_(torch.from_numpy(b))
        self.orig_linear.requires_grad = self.args.train_linear

    def forward(self, inputs):

        raise NotImplementedError
