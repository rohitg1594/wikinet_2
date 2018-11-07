# Mention words and full document context around mention with a combining linear layer
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from src.models.combined.base import CombinedBase
from src.models.loss import Loss

import numpy as np
np.set_printoptions(threshold=10**8)


class FullContext(CombinedBase, Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Mention embeddings
        self.mention_embs = nn.Embedding(self.word_embs.weight.shape[0], self.args.mention_word_dim,
                                         padding_idx=0, sparse=self.args.sparse)
        self.mention_embs.weight.data.normal_(0, self.args.init_stdv)
        self.mention_embs.weight.data[0] = 0

        # Entity mention embeddings
        self.ent_mention_embs = nn.Embedding(self.ent_embs.weight.shape[0], self.args.ent_mention_dim,
                                             padding_idx=0, sparse=self.args.sparse)
        self.ent_mention_embs.weight.data.normal_(0, self.args.init_stdv)
        self.ent_mention_embs.weight.data[0] = 0

        # Linear
        if self.args.combined_linear:
            self.combine_linear = nn.Linear(self.args.mention_word_dim + self.args.context_word_dim,
                                            self.args.mention_word_dim + self.args.context_word_dim)

        # Dropout
        self.dp = nn.Dropout(self.args.dp)

    def forward(self, inputs):
        mention_word_tokens, candidate_ids, context_tokens = inputs

        # Get the embeddings
        mention_embs = self.mention_embs(mention_word_tokens)
        context_embs = self.word_embs(context_tokens)
        candidate_mention_embs = self.ent_mention_embs(candidate_ids)
        candidate_context_embs = self.ent_embs(candidate_ids)

        # Sum the embeddings over the small and large tokens dimension
        mention_embs_agg = torch.mean(mention_embs, dim=1)
        context_embs_agg = torch.mean(context_embs, dim=1)

        # Cat the embs
        mention_repr = torch.cat((mention_embs_agg, context_embs_agg), dim=1)
        cand_repr = torch.cat((candidate_mention_embs, candidate_context_embs), dim=1)
        if self.args.combined_linear:
            mention_repr = self.combine_linear(mention_repr)

        # Normalize
        if self.args.norm_final:
            candidate_embs = F.normalize(cand_repr, dim=1)
            mention_repr = F.normalize(mention_repr, dim=1)

        # Dot product over last dimension only during training
        if len(candidate_ids.shape) == 2:
            mention_repr.unsqueeze_(1)
            scores = torch.matmul(mention_repr, cand_repr.transpose(1, 2)).squeeze(1)
        else:
            scores = torch.Tensor([0])

        return scores, cand_repr, mention_repr

    def loss(self, scores, labels):
        return self.cross_entropy(scores, labels)

