# Model for pre-training -- predicts entity from whole abstract context
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase
from src.models.loss import Loss

from logging import getLogger

logger = getLogger(__name__)


class PreTrainContext(CombinedBase, Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Linear
        if self.args.combined_linear:
            if self.orig_linear.weight.shape == (self.args.context_word_dim, self.args.mention_ent_dim):
                logger.info("Using original linear layer ")
                self.combine_linear = self.orig_linear
            else:
                logger.info("Using new linear layer ")
                self.combine_linear = nn.Linear(self.args.context_word_dim, self.args.mention_ent_dim)

        # Dropout
        self.dp = nn.Dropout(self.args.dp)

    def forward(self, inputs):
        context_tokens = inputs['context_tokens']
        candidate_ids = inputs['candidate_ids']

        # Get the embeddings
        context_repr = self.word_embs(context_tokens)
        candidate_repr = self.ent_combined_embs(candidate_ids)

        # Normalize
        if self.args.norm_final:
            candidate_repr = F.normalize(candidate_repr, dim=1)
            context_repr = F.normalize(context_repr, dim=1)

        # Sum the embeddings / pass through linear
        context_repr = torch.mean(context_repr, dim=1)
        if self.args.combined_linear:
            context_repr = self.orig_linear(context_repr)

        # Dot product over last dimension only during training
        if len(candidate_ids.shape) == 2:
            context_repr.unsqueeze_(1)
            scores = torch.matmul(context_repr, candidate_repr.transpose(1, 2)).squeeze(1)
        else:
            scores = 0

        return scores, candidate_repr, context_repr

    def loss(self, scores, labels):
        return self.cross_entropy(scores, labels)
