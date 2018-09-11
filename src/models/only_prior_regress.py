# Model that only tries to learn prior probabilities, also has regression loss
import torch
import torch.nn.functional as F
import torch.nn as nn

import sys

from src.models.combined.base import CombinedBase


class OnlyPrior(CombinedBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unpack args
        mention_embs = kwargs['mention_embs']
        ent_mention_embs = kwargs['ent_mention_embs']
        ent_prior_embs = kwargs['ent_prior_embs']

        # Mention embeddings
        self.mention_embs = nn.Embedding(*mention_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.mention_embs.weight.data.copy_(mention_embs)

        # Entity mention embeddings
        self.ent_mention_embs = nn.Embedding(*ent_mention_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.ent_mention_embs.weight.data.copy_(ent_mention_embs)

        # Entity prior embeddings
        self.ent_prior_embs = nn.Embedding(*ent_prior_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.ent_prior_embs.weight.data.copy_(ent_prior_embs)

    def forward(self, inputs):
        mention_word_tokens, candidate_ids = inputs

        # print('INPUT TO MODEL')
        #
        # print('MENTION_WORD_TOKENS')
        # print(mention_word_tokens[:5])
        # print('CANDIDATE_IDS')
        # print(candidate_ids[:5, :10])
        # sys.exit(1)

        num_abst, num_ent, num_word = mention_word_tokens.shape
        num_abst, num_ent, num_cand = candidate_ids.shape

        # Reshape to two dimensions - needed because nn.Embedding only allows lookup with 2D-Tensors
        mention_word_tokens = mention_word_tokens.view(-1, num_word)
        candidate_ids = candidate_ids.view(-1, num_cand)

        # Get the embeddings
        mention_embs = self.mention_embs(mention_word_tokens)
        candidate_mention_embs = self.ent_mention_embs(candidate_ids)
        candidate_prior_embs = self.ent_prior_embs(candidate_ids)

        # Sum the embeddings over the small and large tokens dimension
        mention_embs_agg = torch.mean(mention_embs, dim=1)

        # Normalize
        if self.args.norm_final:
            candidate_mention_embs = F.normalize(candidate_mention_embs, dim=2)
            candidate_prior_embs = F.normalize(candidate_prior_embs, dim=2)
            mention_embs_agg = F.normalize(mention_embs_agg, dim=1)

        mention_embs_agg.unsqueeze_(1)

        # Dot product over last dimension
        ranking_scores = (mention_embs_agg * candidate_mention_embs).sum(dim=2)
        regress_scores = (mention_embs_agg * candidate_prior_embs).sum(dim=2)

        return ranking_scores, regress_scores