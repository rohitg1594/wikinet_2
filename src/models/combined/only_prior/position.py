# Model that only tries to learn the prior probability through mention words with positional encodings
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase


class Position(CombinedBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unpack args
        mention_embs = kwargs['mention_embs']
        ent_mention_embs = kwargs['ent_mention_embs']

        # Mention embeddings
        self.mention_embs = nn.Embedding(*mention_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.mention_embs.weight.data.copy_(mention_embs)

        # Entity mention embeddings
        self.ent_mention_embs = nn.Embedding(*ent_mention_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.ent_mention_embs.weight.data.copy_(ent_mention_embs)

        # Positional Encoding
        self.position_lin = nn.Linear(mention_embs.shape[1], mention_embs.shape[1])
        self.position_embs = nn.Embedding(self.args.max_word_size + 1, mention_embs.shape[1], padding_idx=0)

    def forward(self, inputs):
        mention_word_tokens, mention_pos_tokens, candidate_ids = inputs

        if len(mention_word_tokens.shape) == 3:
            num_abst, num_ent, num_word = mention_word_tokens.shape
            num_abst, num_ent, num_cand = candidate_ids.shape

            # Reshape to two dimensions - needed because nn.Embedding only allows lookup with 2D-Tensors
            mention_word_tokens = mention_word_tokens.view(-1, num_word)
            mention_pos_tokens = mention_pos_tokens.view(-1, num_word)
            candidate_ids = candidate_ids.view(-1, num_cand)

            # Get the embeddings
            mention_embs = self.mention_embs(mention_word_tokens)
            candidate_embs = self.ent_mention_embs(candidate_ids)

            # Add pos embs / pass through linear
            mention_pos_embs = self.position_embs(mention_pos_tokens)
            mention_embs_agg = torch.mean(self.position_lin(mention_embs + mention_pos_embs), dim=1)

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=2)
                mention_embs_agg = F.normalize(mention_embs_agg, dim=1)

            mention_embs_agg.unsqueeze_(1)

            # Dot product over last dimension
            scores = (mention_embs_agg * candidate_embs).sum(dim=2)

            return scores

        else:
            # Get the embeddings
            mention_embs = self.mention_embs(mention_word_tokens)
            candidate_embs = self.ent_mention_embs(candidate_ids)

            # Add pos embs / pass through linear
            mention_pos_embs = self.position_embs(mention_pos_tokens)
            mention_embs_agg = torch.mean(self.position_lin(mention_embs + mention_pos_embs), dim=1)

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=1)
                mention_embs_agg = F.normalize(mention_embs_agg, dim=1)

            return candidate_embs, mention_embs_agg
