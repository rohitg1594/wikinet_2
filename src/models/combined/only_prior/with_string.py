# Model that only tries to learn the prior probability through mention words and gram features
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase


class WithString(CombinedBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unpack args
        mention_embs = kwargs['mention_embs']
        ent_mention_embs = kwargs['ent_mention_embs']
        gram_embs = kwargs['gram_embs']

        # Mention embeddings
        self.mention_embs = nn.Embedding(*mention_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.mention_embs.weight.data.copy_(mention_embs)

        # Entity mention embeddings
        self.ent_mention_embs = nn.Embedding(*ent_mention_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.ent_mention_embs.weight.data.copy_(ent_mention_embs)

        # Linear
        self.lin = nn.Linear(mention_embs.shape[1] + gram_embs.shape[1], mention_embs.shape[1] + gram_embs.shape[1],
                             bias=False)
        torch.nn.init.eye(self.lin.weight)

    def forward(self, inputs):
        mention_word_tokens, mention_gram_tokens, candidate_gram_tokens, candidate_ids = inputs

        if len(mention_word_tokens.shape) == 3:
            num_abst, num_ent, num_word = mention_word_tokens.shape
            num_abst, num_ent, num_cand, num_gram = candidate_gram_tokens.shape

            # Reshape to two dimensions - needed because nn.Embedding only allows lookup with 2D-Tensors
            mention_word_tokens = mention_word_tokens.view(-1, num_word)
            mention_gram_tokens = mention_gram_tokens.view(-1, num_gram)
            candidate_ids = candidate_ids.view(-1, num_cand)
            candidate_gram_tokens = candidate_gram_tokens.view(-1, num_gram)

            # Get the embeddings
            mention_embs = self.mention_embs(mention_word_tokens)
            mention_gram_embs = self.gram_embs(mention_gram_tokens)
            candidate_embs = self.ent_mention_embs(candidate_ids)
            candidate_gram_embs = self.gram_embs(candidate_gram_tokens)

            # Reshape candidate gram embs
            candidate_gram_embs = candidate_gram_embs.view(num_abst * num_ent, num_cand, num_gram, -1)

            # Sum the embeddings over the small and large tokens dimension
            mention_embs = torch.mean(mention_embs, dim=1)
            mention_gram_embs = torch.mean(mention_gram_embs, dim=1)
            candidate_gram_embs = torch.mean(candidate_gram_embs, dim=2)

            # Pass through linear layer
            mention_embs_agg = self.lin(torch.cat((mention_embs, mention_gram_embs), dim=1))
            candidate_embs_agg = self.lin(torch.cat((candidate_embs, candidate_gram_embs), dim=2), dim=2)

            # Normalize
            if self.args.norm_final:
                candidate_embs_agg = F.normalize(candidate_embs_agg, dim=2)
                mention_embs_agg = F.normalize(mention_embs_agg, dim=1)

            mention_embs_agg.unsqueeze_(1)

            # Dot product over last dimension
            scores = (mention_embs_agg * candidate_embs_agg).sum(dim=2)

            return scores

        else:
            print('MENTION GRAM TOKENS : {}'.format(mention_gram_tokens[:10, :10]))
            print('CAND GRAM TOKENS : {}'.format(candidate_gram_tokens[:10, :10]))
            
            # Get the embeddings
            mention_embs = self.mention_embs(mention_word_tokens)
            mention_gram_embs = self.gram_embs(mention_gram_tokens)
            candidate_embs = self.ent_mention_embs(candidate_ids)
            candidate_gram_embs = self.gram_embs(candidate_gram_tokens)

            # Sum the embeddings over the small and large tokens dimension
            mention_embs = torch.mean(mention_embs, dim=1)
            mention_gram_embs = torch.mean(mention_gram_embs, dim=1)
            candidate_gram_embs = torch.mean(candidate_gram_embs, dim=1)

            # Pass through linear layer
            mention_embs_agg = self.lin(torch.cat((mention_embs, mention_gram_embs), dim=1))
            candidate_embs_agg = self.lin(torch.cat((candidate_embs, candidate_gram_embs), dim=1), dim=1)

            # Normalize
            if self.args.norm_final:
                candidate_embs_agg = F.normalize(candidate_embs_agg, dim=1)
                mention_embs_agg = F.normalize(mention_embs_agg, dim=1)

            return candidate_embs_agg, mention_embs_agg
