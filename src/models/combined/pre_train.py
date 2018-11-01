# Model for pre-training -- predicts entity from whole abstract context
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase
from src.models.loss import Loss


class PreTrain(CombinedBase, Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # # Context embeddings
        # self.context_embs = nn.Embedding(self.word_embs.weight.shape[0], self.args.context_word_dim,
        #                                  padding_idx=0, sparse=self.args.sparse)
        # self.context_embs.weight.data.normal_(0, self.args.init_stdv)
        # self.context_embs.weight.data[0] = 0
        #
        # # Entity mention embeddings
        # self.ent_mention_embs = nn.Embedding(self.ent_embs.weight.shape[0], self.args.ent_mention_dim,
        #                                      padding_idx=0, sparse=self.args.sparse)
        # self.ent_mention_embs.weight.data.normal_(0, self.args.init_stdv)
        # self.ent_mention_embs.weight.data[0] = 0

        # Linear
        if self.args.combined_linear:
            self.combine_linear = nn.Linear(self.args.context_word_dim, self.args.ent_mention_dim)

        # Dropout
        self.dp = nn.Dropout(self.args.dp)

    def forward(self, inputs):
        context_tokens, candidate_ids = inputs

        # Get the embeddings
        context_repr = self.word_embs(context_tokens)
        candidate_repr = self.ent_embs(candidate_ids)
        print(f'context :  {context_repr.shape}, candidate : {candidate_repr.shape}')

        # Sum the embeddings / pass through linear
        context_repr = torch.mean(context_repr, dim=1)
        if self.args.combined_linear:
            context_repr = self.combine_linear(context_repr)

        # Normalize
        if self.args.norm_final:
            candidate_repr = F.normalize(candidate_repr, dim=1)
            context_repr = F.normalize(context_repr, dim=1)

        # Dot product over last dimension only during training
        if len(candidate_ids.shape) == 2:
            context_repr.unsqueeze_(1)
            scores = torch.matmul(context_repr, candidate_repr.transpose(1, 2)).squeeze(1)
        else:
            scores = 0

        return scores, candidate_repr, context_repr

    def loss(self, scores, labels):
        return self.cross_entropy(scores, labels)
