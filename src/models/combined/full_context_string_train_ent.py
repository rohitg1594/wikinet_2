# Linear Layer to get weights for each component of mention representation
import torch
import torch.nn.functional as F
import torch.nn as nn

from logging import getLogger
from os.path import join

from src.models.combined.base import CombinedBase
from src.models.combined.string_autoencoder import StringAutoEncoder
from src.models.loss import Loss
from src.utils.utils import np_to_tensor

import numpy as np
np.set_printoptions(threshold=10**8)

logger = getLogger(__name__)


class FullContextStringTrainEnt(CombinedBase, Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unpack args
        mention_word_embs = kwargs['mention_word_embs']
        autoencoder_state_dict = kwargs['autoencoder_state_dict']
        hidden_size = kwargs['hidden_size']
        char_embs = kwargs['char_embs']
        total_dims = self.args.mention_word_dim + self.args.context_word_dim + hidden_size
        ent_embs = torch.load(join(self.args.data_path, 'ent_combined_embs.pickle'))

        del self.ent_embs

        # Mention embeddings
        self.mention_word_embs = nn.Embedding(*mention_word_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.mention_word_embs.weight.data.copy_(np_to_tensor(mention_word_embs))
        self.mention_word_embs.requires_grad = self.args.train_mention

        # Entity mention embeddings
        self.ent_embs = nn.Embedding(*ent_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.ent_embs.weight.data.copy_(np_to_tensor(ent_embs))

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

        ##### DAN #######



    def forward(self, inputs):
        mention_word_tokens = inputs['mention_word_tokens']
        mention_char_tokens = inputs['mention_char_tokens']
        candidate_ids = inputs['candidate_ids']
        candidate_char_tokens = inputs['candidate_char_tokens']
        context_tokens = inputs['context_tokens']

        # Get string reps
        _, mention_str_rep, _ = self.autoencoder(mention_char_tokens)

        # Get the embeddings
        mention_embs = self.dp(self.mention_word_embs(mention_word_tokens))
        context_embs = self.dp(self.word_embs(context_tokens))
        candidate_embs = self.dp(self.ent_embs(candidate_ids))


        # Sum the embeddings over the small and large tokens dimension
        mention_embs_agg = torch.mean(mention_embs, dim=1)
        context_embs_agg = self.orig_linear(torch.mean(context_embs, dim=1))

        # Normalize
        mention_embs_agg = F.normalize(mention_embs_agg, dim=len(mention_embs_agg.shape) - 1)
        context_embs_agg = F.normalize(context_embs_agg, dim=len(context_embs_agg.shape) - 1)
        mention_str_rep = F.normalize(mention_str_rep, dim=len(mention_str_rep.shape) - 1)

        # Cat the embs
        cat_dim = 2 if len(candidate_ids.shape) == 2 else 1
        mention_cat = torch.cat((mention_embs_agg, context_embs_agg, mention_str_rep), dim=1)


        # Dot product over last dimension only during training
        if len(candidate_ids.shape) == 2:
            mention_repr.unsqueeze_(1)
            scores = torch.matmul(mention_repr, cand_repr.transpose(1, 2)).squeeze(1)
        else:
            scores = torch.Tensor([0])

        return scores, cand_repr, mention_repr

    def loss(self, scores, labels):
        return self.cross_entropy(scores, labels)
