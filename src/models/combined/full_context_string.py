# Mention words, full document context around mention with a combining linear layer and string info with autoencoder
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase
from src.models.combined.string_autoencoder import StringAutoEncoder
from src.models.loss import Loss

import numpy as np
np.set_printoptions(threshold=10**8)


class FullContextString(CombinedBase, Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Mention embeddings
        self.mention_embs = nn.Embedding(self.word_embs.weight.shape[0], self.args.mention_word_dim,
                                         padding_idx=0, sparse=self.args.sparse)
        self.mention_embs.weight.data.normal_(0, self.args.init_stdv)
        self.mention_embs.weight.data[0] = 0

        # Entity mention embeddings
        self.ent_mention_embs = nn.Embedding(self.ent_embs.weight.shape[0], self.args.ent_mention_dim,
                                             padding_idx=0, sparse=self.args.sparse)
        self.ent_mention_embs.weight.data.normal_(0, self.args.init_stdv)
        self.ent_mention_embs.weight.data[0] = 0

        # Dropout
        self.dp = nn.Dropout(self.args.dp)

        ##### Autoencoder #####
        autoencoder_state_dict = kwargs['autoencoder_state_dict']
        hidden_size = kwargs['hidden_size']
        max_char_size = self.args.max_char_size
        char_embs = kwargs['char_embs']

        self.autoencoder = StringAutoEncoder(max_char_size=max_char_size, hidden_size=hidden_size, char_embs=char_embs)
        self.autoencoder.load_state_dict(autoencoder_state_dict)
        self.autoencoder.requires_grad = False

        # Linear
        if self.args.combined_linear:
            self.combine_linear = nn.Linear(self.args.mention_word_dim + self.args.context_word_dim + hidden_size,
                                            self.args.mention_word_dim + self.args.context_word_dim + hidden_size)

    def forward(self, inputs):
        mention_word_tokens = inputs['mention_word_tokens']
        mention_char_tokens = inputs['mention_char_tokens']
        candidate_ids = inputs['candidate_ids']
        candidate_char_tokens = inputs['candidate_char_tokens']
        context_tokens = inputs['context_tokens']

        # Get string reps
        _, mention_str_rep, _ = self.autoencoder(mention_char_tokens)
        _, candidate_str_rep, _ = self.autoencoder(candidate_char_tokens)

        # Get the embeddings
        mention_embs = self.mention_embs(mention_word_tokens)
        context_embs = self.word_embs(context_tokens)
        candidate_mention_embs = self.ent_mention_embs(candidate_ids)
        candidate_context_embs = self.ent_embs(candidate_ids)

        # Sum the embeddings over the small and large tokens dimension
        mention_embs_agg = torch.mean(mention_embs, dim=1)
        context_embs_agg = F.normalize(self.orig_linear(torch.mean(context_embs, dim=1)))

        # Cat the embs
        cat_dim = 2 if len(candidate_ids.shape) == 2 else 1
        mention_repr = torch.cat((mention_embs_agg, context_embs_agg, mention_str_rep), dim=1)
        cand_repr = torch.cat((candidate_mention_embs, candidate_context_embs, candidate_str_rep), dim=cat_dim)

        if self.args.combined_linear:
            mention_repr = self.combine_linear(mention_repr)

        # Normalize
        if self.args.norm_final:
            cand_repr = F.normalize(cand_repr, dim=cat_dim)
            mention_repr = F.normalize(mention_repr, dim=1)

        # Dot product over last dimension only during training
        if len(candidate_ids.shape) == 2:
            mention_repr.unsqueeze_(1)
            scores = torch.matmul(mention_repr, cand_repr.transpose(1, 2)).squeeze(1)
        else:
            scores = torch.Tensor([0])

        return scores, cand_repr, mention_repr

    def loss(self, scores, labels):
        return self.cross_entropy(scores, labels)
