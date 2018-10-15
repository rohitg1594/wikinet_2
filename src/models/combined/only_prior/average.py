# Model that only tries to learn the prior probability through mention words
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase
from src.models.loss import Loss


class Average(CombinedBase, Loss):

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

    def forward(self, inputs):
        mention_word_tokens, candidate_ids = inputs

        # Get the embeddings
        mention_embs = self.mention_embs(mention_word_tokens)
        candidate_embs = self.ent_mention_embs(candidate_ids)

        # Sum the embeddings over the small and large tokens dimension
        mention_repr = torch.mean(mention_embs, dim=1)

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
