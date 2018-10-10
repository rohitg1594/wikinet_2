# Mention words and small context around mention with a combining linear layer
import torch
import torch.nn.functional as F
import torch.nn as nn

import sys

from src.models.combined.base import CombinedBase


class SmallContext(CombinedBase):

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
        if self.args.combined_linear:
            self.combine_linear = nn.Linear(self.args.mention_word_dim + self.args.context_word_dim,
                                            self.args.ent_mention_dim)

    def forward(self, inputs):
        mention_word_tokens, candidate_ids, context_tokens = inputs

        if len(mention_word_tokens.shape) == 3:
            num_abst, num_ent, num_word = mention_word_tokens.shape
            num_abst, num_ent, num_cand = candidate_ids.shape
            num_abst, num_ent, num_context = context_tokens.shape

            # Reshape to two dimensions - needed because nn.Embedding only allows lookup with 2D-Tensors
            mention_word_tokens = mention_word_tokens.view(-1, num_word)
            candidate_ids = candidate_ids.view(-1, num_cand)
            context_tokens = context_tokens.view(-1, num_context)

            # Get the embeddings
            mention_embs = self.mention_embs(mention_word_tokens)
            context_embs = self.context_embs(context_tokens)
            candidate_embs = self.ent_mention_embs(candidate_ids)

            # Sum the embeddings over the small and large tokens dimension
            mention_embs_agg = torch.mean(mention_embs, dim=1)
            context_embs_agg = torch.mean(context_embs, dim=1)

            # Cat the embs / pass through linear layer
            mention_repr = torch.cat((mention_embs_agg, context_embs_agg), dim=1)
            if self.args.combined_linear:
                mention_repr = self.combine_linear(mention_repr)

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=2)
                mention_repr = F.normalize(mention_repr, dim=1)

            mention_repr.unsqueeze_(1)

            # Dot product over last dimension
            scores = torch.matmul(mention_repr, candidate_embs.transpose(1, 2)).squeeze(1)

            return scores

        else:
            # Get the embeddings
            mention_embs = self.mention_embs(mention_word_tokens)
            context_embs = self.context_embs(context_tokens)
            candidate_embs = self.ent_mention_embs(candidate_ids)

            # Sum the embeddings over the small and large tokens dimension
            mention_embs_agg = torch.mean(mention_embs, dim=1)
            context_embs_agg = torch.mean(context_embs, dim=1)

            # Cat the embs
            mention_repr = torch.cat((mention_embs_agg, context_embs_agg), dim=1)
            if self.args.combine_linear:
                mention_repr = self.combine_linear(mention_repr)

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=1)
                mention_repr = F.normalize(mention_repr, dim=1)

            return candidate_embs, mention_repr
