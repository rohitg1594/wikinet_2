# Model that only uses context and gram information. It concatenates gram and context information by weighing them first.
import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from src.models.combined.combined_base import CombinedBase


class CombinedContextGramWeighted(CombinedBase):

    def __init__(self, word_embs=None, ent_embs=None, W=None, b=None, gram_embs=None, args=None):
        super().__init__(word_embs, ent_embs, W, b, gram_embs, args)

        self.weighing_linear_ent = nn.Linear(ent_embs.shape[1] + gram_embs.shape[1], 1, bias=False)
        self.weighing_linear_ent.weight.data.copy_(torch.from_numpy(np.ones((ent_embs.shape[1] + gram_embs.shape[1]))))
        self.weighing_linear_mention = nn.Linear(ent_embs.shape[1] + gram_embs.shape[1], 1, bias=False)
        self.weighing_linear_mention.weight.data.copy_(torch.from_numpy(np.ones((ent_embs.shape[1] + gram_embs.shape[1]))))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        mention_gram_tokens, context_word_tokens, candidate_gram_tokens, candidate_ids = inputs

        num_abst, num_ent, num_cand, num_gram = candidate_gram_tokens.shape
        num_abst, num_ent, num_context = context_word_tokens.shape

        # Reshape to two dimensions - needed because nn.Embedding only allows lookup with 2D-Tensors
        mention_gram_tokens = mention_gram_tokens.view(-1, num_gram)
        context_word_tokens = context_word_tokens.view(-1, num_context)
        candidate_gram_tokens = candidate_gram_tokens.view(-1, num_gram)
        candidate_ids = candidate_ids.view(-1, num_cand)

        # Get the embeddings
        mention_gram_embs = self.gram_embs(mention_gram_tokens)
        candidate_gram_embs = self.gram_embs(candidate_gram_tokens)
        context_word_embs = self.word_embs(context_word_tokens)
        candidate_ent_embs = self.ent_embs(candidate_ids)

        # Reshape to original shape
        candidate_gram_embs = candidate_gram_embs.view(num_abst * num_ent, num_cand, num_gram, -1)

        # Sum the embeddings over the small and large tokens dimension
        mention_gram_embs = torch.mean(mention_gram_embs, dim=1)
        candidate_gram_embs = torch.mean(candidate_gram_embs, dim=2)

        mention_gram_embs = F.normalize(mention_gram_embs, dim=1)
        candidate_gram_embs = F.normalize(candidate_gram_embs, dim=2)

        context_word_embs = torch.mean(context_word_embs, dim=1)
        context_word_embs = self.orig_linear(context_word_embs)

        context_word_embs = F.normalize(context_word_embs, dim=1)

        # Concatenate word / gram embeddings (unweighted)
        combined_ent_unw = torch.cat((candidate_ent_embs, candidate_gram_embs), dim=2)
        combined_mention_unw = torch.cat((context_word_embs, mention_gram_embs), dim=1)

        # Calculate weights
        w_ent = self.sigmoid(self.weighing_linear_ent(combined_ent_unw))
        w_mention = self.sigmoid(self.weighing_linear_mention(combined_mention_unw))

        # Concatenate word / gram embeddings (weighted)
        combined_ent_w = torch.cat((w_ent * candidate_ent_embs, (1 - w_ent) * candidate_gram_embs), dim=2)
        combined_mention_w = torch.cat((w_mention * context_word_embs, (1 - w_mention) * mention_gram_embs), dim=1)
        combined_mention_w.unsqueeze_(1)

        # Dot product over last dimension
        scores = (combined_mention_w * combined_ent_w).sum(dim=2)

        return scores