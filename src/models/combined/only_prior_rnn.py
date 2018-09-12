# Only prior model with LSTM
import torch.nn.functional as F
import torch.nn as nn
import torch

from src.models.combined.base import CombinedBase


class OnlyPriorRNN(CombinedBase):

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

        # Mention linear
        self.lstm = nn.LSTM(mention_embs.shape[1], mention_embs.shape[1], 2, batch_first=True)

    def forward(self, inputs):
        mention_word_tokens, candidate_ids = inputs

        if len(mention_word_tokens.shape) == 3:
            num_abst, num_ent, num_word = mention_word_tokens.shape
            num_abst, num_ent, num_cand = candidate_ids.shape

            # Reshape to two dimensions - needed because nn.Embedding only allows lookup with 2D-Tensors
            mention_word_tokens = mention_word_tokens.view(-1, num_word)
            candidate_ids = candidate_ids.view(-1, num_cand)

            # Mask for lstm
            mask = (mention_word_tokens > 0).sum(1)

            # Get the embeddings
            mention_embs = self.mention_embs(mention_word_tokens)
            candidate_embs = self.ent_mention_embs(candidate_ids)

            # Sum the embeddings over the small and large tokens dimension
            mention_embs, _ = self.lstm(mention_embs)
            mention_embs = mention_embs[torch.arange(mention_embs.shape[0]).long(), mask - 1]

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=2)
                mention_embs = F.normalize(mention_embs, dim=1)

            mention_embs.unsqueeze_(1)

            # Dot product over last dimension
            scores = (mention_embs * candidate_embs).sum(dim=2)

            return scores

        else:
            # Mask for lstm
            mask = (mention_word_tokens > 0).sum(1)

            # Get the embeddings
            mention_embs = self.mention_embs(mention_word_tokens)
            candidate_embs = self.ent_mention_embs(candidate_ids)

            # Sum the embeddings over the small and large tokens dimension
            mention_embs, _ = self.lstm(mention_embs)
            mention_embs = mention_embs[torch.arange(mention_embs.shape[0]).long(), mask - 1]

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=1)
                mention_embs = F.normalize(mention_embs, dim=1)

            return candidate_embs, mention_embs
