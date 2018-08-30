# Model that only uses context and gram information. It concatenates gram and context information by weighing them first.
import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from src.models.combined.combined_base import CombinedBase


class CombinedContextGramWeighted(CombinedBase):

    def __init__(self, word_embs=None, ent_embs=None, W=None, b=None, gram_embs=None, args=None):
        super().__init__(word_embs, ent_embs, W, b, gram_embs, args)

        self.weighing_linear = nn.Linear(ent_embs.shape[1] + gram_embs.shape[1], 1, bias=False)
        self.weighing_linear.weight.data.copy_(torch.from_numpy(np.ones((ent_embs.shape[1] + gram_embs.shape[1]))))
        self.sigmoid = nn.Sigmoid()
        self.dp = nn.Dropout(0.3)

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

        # Apply Dropout
        mention_gram_embs = self.dp(mention_gram_embs)
        candidate_gram_embs = self.dp(candidate_gram_embs)
        context_word_embs = self.dp(context_word_embs)
        candidate_ent_embs = self.dp(candidate_ent_embs)

        # Reshape to original shape
        candidate_gram_embs = candidate_gram_embs.view(num_abst * num_ent, num_cand, num_gram, -1)

        # Sum the embeddings over the small and large tokens dimension
        mention_gram_embs = torch.mean(mention_gram_embs, dim=1)
        candidate_gram_embs = torch.mean(candidate_gram_embs, dim=2)
        context_word_embs = torch.mean(context_word_embs, dim=1)
        context_word_embs = self.orig_linear(context_word_embs)

        # Normalize
        mention_gram_embs = F.normalize(mention_gram_embs, dim=1)
        candidate_gram_embs = F.normalize(candidate_gram_embs, dim=2)
        context_word_embs = F.normalize(context_word_embs, dim=1)

        # Concatenate word / gram embeddings (unweighted)
        combined_ent = torch.cat((candidate_ent_embs, candidate_gram_embs), dim=2)
        combined_mention_unw = torch.cat((context_word_embs, mention_gram_embs), dim=1)

        # Calculate weights
        w = self.sigmoid(self.weighing_linear(combined_mention_unw))

        # Apply dropout
        w = self.dp(w)

        # Concatenate word / gram embeddings (weighted)
        combined_mention_w = torch.cat((w * context_word_embs, (1 - w) * mention_gram_embs), dim=1)
        combined_mention_w.unsqueeze_(1)

        # Dot product over last dimension
        scores = (combined_mention_w * combined_ent).sum(dim=2)

        return scores
