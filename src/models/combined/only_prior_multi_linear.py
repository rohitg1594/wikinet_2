# Only prior model with multiple linear layers
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase


class OnlyPriorMultiLinear(CombinedBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unpack args
        mention_embs = kwargs['mention_embs']
        ent_mention_embs = kwargs['ent_mention_embs']

        # Mention embeddings
        self.mention_embs = nn.Embedding(mention_embs.shape[0], mention_embs.shape[1], padding_idx=0, sparse=self.args.sparse)
        self.mention_embs.weight.data.copy_(mention_embs)

        # Entity mention embeddings
        self.ent_mention_embs = nn.Embedding(ent_mention_embs.shape[0], ent_mention_embs.shape[1], padding_idx=0, sparse=self.args.sparse)
        self.ent_mention_embs.weight.data.copy_(ent_mention_embs)

        # Mention linear
        self.mention_linear1 = nn.Linear(ent_mention_embs.shape[1], 128)
        self.mention_linear2 = nn.Linear(128, 128)
        self.mention_linear3 = nn.Linear(128, ent_mention_embs.shape[1])

    def forward(self, inputs):
        mention_word_tokens, candidate_ids = inputs

        if len(mention_word_tokens.shape) == 3:
            num_abst, num_ent, num_word = mention_word_tokens.shape
            num_abst, num_ent, num_cand = candidate_ids.shape

            # Reshape to two dimensions - needed because nn.Embedding only allows lookup with 2D-Tensors
            mention_word_tokens = mention_word_tokens.view(-1, num_word)
            candidate_ids = candidate_ids.view(-1, num_cand)

            # Get the embeddings
            mention_embs = self.mention_embs(mention_word_tokens)
            candidate_embs = self.ent_mention_embs(candidate_ids)

            # Sum the embeddings over the small and large tokens dimension
            mention_embs = torch.mean(mention_embs, dim=1)

            # Transform with linear layer
            mention_embs = self.mention_linear1(mention_embs)
            mention_embs = F.relu(mention_embs)
            mention_embs = self.mention_linear2(mention_embs)
            mention_embs = F.relu(mention_embs)
            mention_embs = self.mention_linear3(mention_embs)

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=2)
                mention_embs = F.normalize(mention_embs, dim=1)

            mention_embs.unsqueeze_(1)

            # Dot product over last dimension
            scores = (mention_embs * candidate_embs).sum(dim=2)

            return scores

        else:
            # Get the embeddings
            mention_embs = self.mention_embs(mention_word_tokens)
            candidate_embs = self.ent_mention_embs(candidate_ids)

            # Sum the embeddings over the small and large tokens dimension
            mention_embs = torch.mean(mention_embs, dim=1)

            # Transform with linear layer
            mention_embs = self.mention_linear1(mention_embs)
            mention_embs = F.relu(mention_embs)
            mention_embs = self.mention_linear2(mention_embs)
            mention_embs = F.relu(mention_embs)
            mention_embs = self.mention_linear3(mention_embs)

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=2)
                mention_embs = F.normalize(mention_embs, dim=1)

            return candidate_embs, mention_embs
