# Full model, weights are learned by three separate linear layers
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase
from src.models.combined.string_autoencoder import StringAutoEncoder
from src.models.loss import Loss
from src.utils.utils import np_to_tensor

import numpy as np
np.set_printoptions(threshold=10**8)


class FullContextStringFromScratchEnt(CombinedBase, Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unpack args
        mention_word_embs = kwargs['mention_word_embs']
        autoencoder_state_dict = kwargs['autoencoder_state_dict']
        hidden_size = kwargs['hidden_size']
        char_embs = kwargs['char_embs']

        # Mention embeddings
        self.mention_word_embs = nn.Embedding(*mention_word_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.mention_word_embs.weight.data.copy_(np_to_tensor(mention_word_embs))
        self.mention_word_embs.requires_grad = self.args.train_mention

        # Dropout
        self.dp = nn.Dropout(self.args.dp)

        ##### Autoencoder #####
        max_char_size = self.args.max_char_size
        self.autoencoder = StringAutoEncoder(max_char_size=max_char_size,
                                             hidden_size=hidden_size,
                                             char_embs=char_embs,
                                             dp=self.args.dp,
                                             activate=self.args.activate)
        self.autoencoder.load_state_dict(autoencoder_state_dict)
        self.autoencoder.requires_grad = False

        # Linear
        self.context_linear = nn.Linear(self.args.mention_word_dim + self.args.context_word_dim + hidden_size,
                                        1, bias=False)
        # nn.init.eye_(self.context_linear.weight)
        self.prior_linear = nn.Linear(self.args.mention_word_dim + self.args.context_word_dim + hidden_size,
                                      1, bias=False)
        # nn.init.eye_(self.prior_linear.weight)
        self.str_linear = nn.Linear(self.args.mention_word_dim + self.args.context_word_dim + hidden_size,
                                      1, bias=False)
        # nn.init.eye_(self.str_linear.weight)

    def forward(self, inputs):
        mention_word_tokens = inputs['mention_word_tokens']
        mention_char_tokens = inputs['mention_char_tokens']
        context_tokens = inputs['context_tokens']
        candidate_ids = inputs['candidate_ids']

        # Get string reps
        _, mention_str_rep, _ = self.autoencoder(mention_char_tokens)

        # Get the embeddings
        mention_embs = self.dp(self.mention_word_embs(mention_word_tokens))
        context_embs = self.dp(self.word_embs(context_tokens))
        candidate_embs = self.dp(self.ent_embs(candidate_ids))

        print(f'CANDIDATE EMBS - {candidate_embs.shape}')

        # Sum the embeddings over the small and large tokens dimension
        mention_embs_agg = torch.mean(mention_embs, dim=1)
        context_embs_agg = self.orig_linear(torch.mean(context_embs, dim=1))

        # Normalize
        mention_embs_agg = F.normalize(mention_embs_agg, dim=len(mention_embs_agg.shape) - 1)
        context_embs_agg = F.normalize(context_embs_agg, dim=len(context_embs_agg.shape) - 1)
        mention_str_rep = F.normalize(mention_str_rep, dim=len(mention_str_rep.shape) - 1)
        cand_repr = F.normalize(candidate_embs, dim=len(candidate_embs.shape) - 1)

        # Cat the embs
        mention_repr = torch.cat((mention_embs_agg, context_embs_agg, mention_str_rep), dim=1)
        print(f'MENTION EMBS - {mention_repr.shape}')
        
        # Get the weights
        mention_weights = self.prior_linear(mention_repr)
        context_weights = self.context_linear(mention_repr)
        str_weights = self.str_linear(mention_repr)

        if self.args.sigmoid:
            mention_weights = nn.Sigmoid(mention_weights)
            context_weights = nn.Sigmoid(context_weights)
            str_weights = nn.Sigmoid(str_weights)

        # Scale mention
        mention_repr_scaled = torch.cat((mention_repr[:, :self.args.mention_word_dim] * mention_weights,
                                         mention_repr[:,
                                         self.args.mention_word_dim: self.args.mention_word_dim + self.args.context_word_dim]
                                         * context_weights,
                                         mention_repr[:, self.args.mention_word_dim + self.args.context_word_dim:]
                                         * str_weights), dim=1)

        # Normalize
        if self.args.norm_final:
            mention_repr_scaled = F.normalize(mention_repr_scaled, dim=1)

        # Dot product over last dimension only during training
        if len(candidate_ids.shape) == 2:
            mention_repr_scaled.unsqueeze_(1)
            scores = torch.matmul(mention_repr_scaled, cand_repr.transpose(1, 2)).squeeze(1)
        else:
            scores = torch.Tensor([0])

        return scores, cand_repr, mention_repr_scaled

    def loss(self, scores, labels):
        return self.cross_entropy(scores, labels)
