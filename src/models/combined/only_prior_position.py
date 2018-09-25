# Model that only tries to learn the prior probability through mention words with positional encodings
import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)

        return emb


class OnlyPriorPosition(CombinedBase):

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

        # Positional Encoding
        self.position_lin = nn.Linear(mention_embs.shape[1], mention_embs.shape[1])
        self.position = PositionalEncoding(self.args.dp, mention_embs.shape[1])

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

            # Add pos embs / pass through linear
            mention_embs_agg = torch.mean(self.position_lin(self.position(mention_embs)), dim=1)

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=2)
                mention_embs_agg = F.normalize(mention_embs_agg, dim=1)

            mention_embs_agg.unsqueeze_(1)

            # Dot product over last dimension
            scores = (mention_embs_agg * candidate_embs).sum(dim=2)

            return scores

        else:
            # Get the embeddings
            mention_embs = self.mention_embs(mention_word_tokens)
            candidate_embs = self.ent_mention_embs(candidate_ids)

            print('MENTION EMBS SHAPE : {}'.format(mention_embs.shape))
            # Sum the embeddings over the small and large tokens dimension
            mention_embs_agg = torch.mean(self.position_lin(self.position(mention_embs)), dim=1)

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=1)
                mention_embs_agg = F.normalize(mention_embs_agg, dim=1)

            return candidate_embs, mention_embs_agg
