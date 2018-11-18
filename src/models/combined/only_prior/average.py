# Model that only tries to learn the prior probability through mention words
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase
from src.models.loss import Loss
from src.utils.utils import np_to_tensor


class Average(CombinedBase, Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unpack args
        mention_word_embs = kwargs['mention_word_embs']
        mention_ent_embs = kwargs['mention_ent_embs']

        # Mention embeddings
        self.mention_word_embs = nn.Embedding(*mention_word_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.mention_word_embs.weight.data.copy_(np_to_tensor(mention_word_embs))

        # Entity mention embeddings
        self.mention_ent_embs = nn.Embedding(*mention_ent_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.mention_ent_embs.weight.data.copy_(np_to_tensor(mention_ent_embs))

        # Dp
        self.do = nn.Dropout(self.args.dp)

    def forward(self, inputs):
        mention_word_tokens = inputs['mention_word_tokens']
        candidate_ids = inputs['candidate_ids']

        # Get the embeddings
        mention_embs = self.dp(self.mention_word_embs(mention_word_tokens))
        candidate_embs = self.dp(self.mention_ent_embs(candidate_ids))

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
