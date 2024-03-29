# Only prior model with Convolution
import torch.nn.functional as F
import torch.nn as nn
import torch

from src.models.combined.base import CombinedBase


class Conv(CombinedBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unpack args
        mention_embs = kwargs['mention_embs']
        ent_mention_embs = kwargs['ent_mention_embs']
        conv_weights = kwargs['conv_weights']

        # Mention embeddings
        self.mention_embs = nn.Embedding(*mention_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.mention_embs.weight.data.copy_(mention_embs)

        # Entity mention embeddings
        self.ent_mention_embs = nn.Embedding(*ent_mention_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.ent_mention_embs.weight.data.copy_(ent_mention_embs)

        # Conv Layer
        if self.args.gram_type == 'bigram':
            kernel = 2
            self.ignore = 1
        else:
            kernel = 3
            self.ignore = 2
        self.conv = torch.nn.Conv1d(in_channels=mention_embs.shape[1], out_channels=mention_embs.shape[1],
                                    kernel_size=kernel, dilation=1, bias=False, padding=1)
        self.conv.weight.data.copy_(conv_weights)

    def forward(self, inputs):
        mention_gram_tokens, candidate_ids = inputs

        if len(mention_gram_tokens.shape) == 3:
            num_abst, num_ent, num_gram = mention_gram_tokens.shape
            num_abst, num_ent, num_cand = candidate_ids.shape

            # Reshape to two dimensions - needed because nn.Embedding only allows lookup with 2D-Tensors
            mention_gram_tokens = mention_gram_tokens.view(-1, num_gram)
            candidate_ids = candidate_ids.view(-1, num_cand)

            # Get the embeddings
            mention_embs = self.mention_embs(mention_gram_tokens).transpose(1, 2)
            candidate_embs = self.ent_mention_embs(candidate_ids)

            # Mask for conv
            mask = (mention_gram_tokens > 0).unsqueeze(1).float()

            # Encode mention embs
            conved_embs = self.conv(mention_embs)[:, :, :-self.ignore]
            conved_embs = conved_embs + mention_embs  # Residual connection
            conved_embs = (conved_embs * mask).sum(dim=2)

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=2)
                conved_embs = F.normalize(conved_embs, dim=1)

            conved_embs.unsqueeze_(1)

            # Dot product over last dimension
            scores = (conved_embs * candidate_embs).sum(dim=2)

            return scores

        else:
            # Get the embeddings
            mention_embs = self.mention_embs(mention_gram_tokens).transpose(1, 2)
            candidate_embs = self.ent_mention_embs(candidate_ids)

            # Mask for conv
            print('MENTION_GRAM_TOKENS : {}'.format(mention_gram_tokens[:10, :30]))
            mask = (mention_gram_tokens > 0).unsqueeze(1).float()
            print('MASK : {}'.format(mask[:10, :30]))

            # Encode mention embs
            conved_embs = self.conv(mention_embs)[:, :, :-self.ignore]
            print('MENTION EMBS SHAPE : {}, CONVED EMBS : {}'.format(mention_embs.shape, conved_embs.shape))
            conved_embs = conved_embs + mention_embs  # Residual connection
            conved_embs = (conved_embs * mask).sum(dim=2)

            # Normalize
            if self.args.norm_final:
                candidate_embs = F.normalize(candidate_embs, dim=1)
                conved_embs = F.normalize(conved_embs, dim=1)

            return candidate_embs, conved_embs
