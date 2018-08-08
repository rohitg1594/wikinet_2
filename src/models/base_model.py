# Other models are based on this model
import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self, yamada_model=None, gram_embs=None, args=None):
        super().__init__()
        self.args = args

        self.word_embs = nn.Embedding(yamada_model['word_embs'].shape[0], yamada_model['word_embs'].shape[1],
                                      padding_idx=0, sparse=self.args.sparse)
        self.word_embs.weight.data.copy_(torch.from_numpy(yamada_model['word_embs']))
        self.word_embs.weight.requires_grad = self.args.train_word

        self.ent_embs = nn.Embedding(yamada_model['ent_embs'].shape[0], yamada_model['ent_embs'].shape[1],
                                     padding_idx=0, sparse=self.args.sparse)
        self.ent_embs.weight.data.copy_(torch.from_numpy(yamada_model['ent_embs']))
        self.ent_embs.weight.requires_grad = self.args.train_ent

        self.orig_linear = nn.Linear(yamada_model['W'].shape[0], yamada_model['W'].shape[1])
        self.orig_linear.weight.data.copy_(torch.from_numpy(yamada_model['W']))
        self.orig_linear.bias.data.copy_(torch.from_numpy(yamada_model['b']))
        self.orig_linear.requires_grad = self.args.train_linear

        self.gram_embs = nn.Embedding(gram_embs.shape[0], gram_embs.shape[1], padding_idx=0, sparse=self.args.sparse)
        self.gram_embs.weight.data.copy_(torch.from_numpy(gram_embs))
        self.gram_embs.weight.requires_grad = self.args.train_gram

    def forward(self, inputs):

        raise NotImplementedError
