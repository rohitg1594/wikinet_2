# Full model, weights are learned by three separate linear layers
import torch
import torch.nn.functional as F
import torch.nn as nn


from src.models.combined.base import CombinedBase
from src.models.combined.string_autoencoder import StringAutoEncoder
from src.models.loss import Loss
from src.utils.utils import np_to_tensor

from logging import getLogger

logger = getLogger(__name__)


class FullContextStringLinearScalar(CombinedBase, Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unpack args
        mention_word_embs = kwargs['mention_word_embs']
        mention_ent_embs = kwargs['mention_ent_embs']
        autoencoder_state_dict = kwargs['autoencoder_state_dict']
        hidden_size = kwargs['hidden_size']
        char_embs = kwargs['char_embs']

        # Mention embeddings
        self.mention_word_embs = nn.Embedding(*mention_word_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.mention_word_embs.weight.data.copy_(np_to_tensor(mention_word_embs))
        self.mention_word_embs.requires_grad = self.args.train_mention

        # Entity mention embeddings
        self.mention_ent_embs = nn.Embedding(*mention_ent_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.mention_ent_embs.weight.data.copy_(np_to_tensor(mention_ent_embs))
        self.mention_ent_embs.requires_grad = self.args.train_mention

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
        self.context_linear = nn.Linear(self.args.mention_word_dim + self.args.context_word_dim + hidden_size, 1)
        self.prior_linear = nn.Linear(self.args.mention_word_dim + self.args.context_word_dim + hidden_size, 1)
        self.str_linear = nn.Linear(self.args.mention_word_dim + self.args.context_word_dim + hidden_size, 1)

        init_func = getattr(nn.init, self.args.init_linear)
        logger.info(f'Initializing Linear layers using {init_func.__name__}')
        init_func(self.context_linear.weight)
        init_func(self.prior_linear.weight)
        init_func(self.str_linear.weight)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        mention_word_tokens = inputs['mention_word_tokens']
        mention_char_tokens = inputs['mention_char_tokens']
        candidate_ids = inputs['candidate_ids']
        candidate_char_tokens = inputs['candidate_char_tokens']
        context_tokens = inputs['context_tokens']

        # Get string reps
        _, mention_str_rep, _ = self.dp(self.autoencoder(mention_char_tokens))
        _, candidate_str_rep, _ = self.autoencoder(candidate_char_tokens)

        # Get the embeddings
        mention_embs = self.dp(self.mention_word_embs(mention_word_tokens))
        context_embs = self.dp(self.word_embs(context_tokens))
        candidate_mention_embs = self.mention_ent_embs(candidate_ids)
        candidate_context_embs = self.ent_embs(candidate_ids)

        # Sum the embeddings over the small and large tokens dimension
        mention_embs_agg = torch.mean(mention_embs, dim=1)
        context_embs_agg = self.orig_linear(torch.mean(context_embs, dim=1))

        # Normalize
        mention_embs_agg = F.normalize(mention_embs_agg, dim=len(mention_embs_agg.shape) - 1)
        candidate_mention_embs = F.normalize(candidate_mention_embs, dim=len(candidate_mention_embs.shape) - 1)
        context_embs_agg = F.normalize(context_embs_agg, dim=len(context_embs_agg.shape) - 1)
        candidate_context_embs = F.normalize(candidate_context_embs, dim=len(candidate_context_embs.shape) - 1)
        mention_str_rep = F.normalize(mention_str_rep, dim=len(mention_str_rep.shape) - 1)
        candidate_str_rep = F.normalize(candidate_str_rep, dim=len(candidate_str_rep.shape) - 1)

        # Cat the embs
        cat_dim = 2 if len(candidate_ids.shape) == 2 else 1
        mention_repr = torch.cat((mention_embs_agg, context_embs_agg, mention_str_rep), dim=1)
        cand_repr = torch.cat((candidate_mention_embs, candidate_context_embs, candidate_str_rep), dim=cat_dim)

        # Get the weights
        mention_weights = self.prior_linear(mention_repr)
        context_weights = self.context_linear(mention_repr)
        str_weights = self.str_linear(mention_repr)

        if self.args.sigmoid:
            mention_weights = self.sigmoid(mention_weights)
            context_weights = self.sigmoid(context_weights)
            str_weights = self.sigmoid(str_weights)

        mention_repr_scaled = torch.cat((mention_repr[:, :self.args.mention_word_dim] * mention_weights,
                                         mention_repr[:, self.args.mention_word_dim: self.args.mention_word_dim + self.args.context_word_dim] * context_weights,
                                         mention_repr[:, self.args.mention_word_dim + self.args.context_word_dim:] * str_weights), dim=1)

        # Normalize
        if self.args.norm_final:
            cand_repr = F.normalize(cand_repr, dim=cat_dim)
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
