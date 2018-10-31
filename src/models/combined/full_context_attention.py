# Mention words and full document context around mention with a combining linear layer
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase
from src.models.loss import Loss

import numpy as np
np.set_printoptions(threshold=10**8)


class FullContextAttn(CombinedBase, Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Mention embeddings
        self.mention_embs = nn.Embedding(self.word_embs.weight.shape[0], self.args.mention_word_dim,
                                         padding_idx=0, sparse=self.args.sparse)
        self.mention_embs.weight.data.normal_(0, self.args.init_stdv)
        self.mention_embs.weight.data[0] = 0

        # Context embeddings
        self.context_embs = nn.Embedding(self.word_embs.weight.shape[0], self.args.context_word_dim,
                                         padding_idx=0, sparse=self.args.sparse)
        self.context_embs.weight.data.normal_(0, self.args.init_stdv)
        self.context_embs.weight.data[0] = 0

        # Entity mention embeddings
        self.ent_mention_embs = nn.Embedding(self.ent_embs.weight.shape[0], self.args.ent_mention_dim,
                                             padding_idx=0, sparse=self.args.sparse)
        self.ent_mention_embs.weight.data.normal_(0, self.args.init_stdv)
        self.ent_mention_embs.weight.data[0] = 0

        # Linear
        self.proj_mention = nn.Linear(self.args.mention_word_dim, self.args.ent_mention_dim)
        self.proj_context = nn.Linear(self.args.context_word_dim, self.args.ent_mention_dim)
        self.w = nn.Parameter(torch.Tensor([0.5]))

        # Dropout
        self.dp = nn.Dropout(self.args.dp)

    def forward(self, inputs):
        mention_word_tokens, candidate_ids, context_tokens = inputs

        # Get the embeddings
        mention_embs = self.mention_embs(mention_word_tokens)
        context_embs = self.mention_embs(context_tokens)
        candidate_embs = self.ent_mention_embs(candidate_ids)

        # Sum the embeddings over the small and large tokens dimension
        mention_embs_agg = torch.mean(mention_embs, dim=1)
        context_embs_agg = torch.mean(context_embs, dim=1)

        # Proj embs
        mention_embs_proj = self.proj_mention(mention_embs_agg)
        context_embs_proj = self.proj_context(context_embs_agg)
        mention_repr = self.w * mention_embs_proj + (1 - self.w) * context_embs_proj

        if len(candidate_ids) > 10**3:
            print(f'combine weight : {self.w}')
        # Normalize
        if self.args.norm_final:
            candidate_embs = F.normalize(candidate_embs, dim=1)
            mention_repr = F.normalize(mention_repr, dim=1)

        # Dot product over last dimension only during training
        if len(candidate_ids.shape) == 2:
            mention_repr.unsqueeze_(1)
            scores = torch.matmul(mention_repr, candidate_embs.transpose(1, 2)).squeeze(1)
        else:
            scores = 0

        return scores, candidate_embs, mention_repr

    def loss(self, scores, labels):
        return self.cross_entropy(scores, labels)

