# Model that only tries to learn the prior probability through mention words
import torch
import torch.nn.functional as F
import torch.nn as nn

import sys

from src.models.combined.base import CombinedBase


class SmallContext(CombinedBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unpack args
        mention_embs = kwargs['mention_embs']
        ent_mention_embs = kwargs['ent_mention_embs']

        # Mention embeddings
        self.mention_embs = nn.Embedding(*mention_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.mention_embs.weight.data.copy_(mention_embs)

        # Context embeddings
        self.context_embs = nn.Embedding(*mention_embs.shape, padding_idx=0, sparse=self.args.sparse)

        # Entity mention embeddings
        self.ent_mention_embs = nn.Embedding(*ent_mention_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.ent_mention_embs.weight.data.copy_(ent_mention_embs)

        # Linear
        self.combine_linear = nn.Linear(2 * mention_embs.shape[1], ent_mention_embs.shape[1])

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
            mention_cat = torch.cat((mention_embs_agg, context_embs_agg), dim=1)
            mention_repr = self.combine_linear(mention_cat)

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=2)
                mention_repr = F.normalize(mention_repr, dim=1)

            mention_repr.unsqueeze_(1)

            # Dot product over last dimension
            scores = (mention_repr * candidate_embs).sum(dim=2)

            return scores

        else:

            # print(f'MENTION WORD TOKENS : {mention_word_tokens[:20]}')
            # print(f'CONTEXT TOKENS : {context_tokens[:20]}')
            # print(f'CANDIDATE IDS : {candidate_ids[:20]}')
            # print(f'LINEAR : {self.combine_linear.weight[:10, :10]}')
            # print(f'MENTION EMBS : {self.mention_embs.weight[:10, :10]}')
            # print(f'CONTEXT EMBS : {self.context_embs.weight[:10, :10]}')
            # print(f'CANDIDATE EMBS : {self.ent_mention_embs.weight[:10, :10]}')

            # Get the embeddings
            mention_embs = self.mention_embs(mention_word_tokens)
            context_embs = self.context_embs(context_tokens)
            candidate_embs = self.ent_mention_embs(candidate_ids)

            # Sum the embeddings over the small and large tokens dimension
            mention_embs_agg = torch.mean(mention_embs, dim=1)
            context_embs_agg = torch.mean(context_embs, dim=1)

            # Cat the embs
            mention_cat = torch.cat((mention_embs_agg, context_embs_agg), dim=1)
            mention_repr = self.combine_linear(mention_cat)

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=1)
                mention_repr = F.normalize(mention_repr, dim=1)

            return candidate_embs, mention_repr
