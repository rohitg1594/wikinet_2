# Model that only uses context, gram and seperate mention and entity embeddings to learn priors
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.combined_base import CombinedBase


class CombinedContextGramMention(CombinedBase):

    def __init__(self, word_embs=None, ent_embs=None, mention_embs=None, ent_mention_embs=None,
                 W=None, b=None, gram_embs=None, args=None):
        super().__init__(word_embs, ent_embs, W, b, gram_embs, args)

        self.mention_embs = nn.Embedding(mention_embs.shape[0], mention_embs.shape[1], padding_idx=0, sparse=self.args.sparse)
        self.mention_embs.weight.data.copy_(torch.from_numpy(mention_embs))
        self.mention_embs.weight.requires_grad = self.args.train_mention

        self.ent_mention_embs = nn.Embedding(ent_mention_embs.shape[0], ent_mention_embs.shape[1], padding_idx=0, sparse=self.args.sparse)
        self.ent_mention_embs.weight.data.copy_(torch.from_numpy(ent_mention_embs))
        self.ent_mention_embs.weight.requires_grad = self.args.train_mention

    def forward(self, inputs):
        mention_gram_tokens, mention_word_tokens, context_word_tokens, candidate_gram_tokens, candidate_word_tokens,\
        candidate_ids = inputs

        num_abst, num_ent, num_cand, num_gram = candidate_gram_tokens.shape
        num_abst, num_ent, num_context = context_word_tokens.shape
        num_abst, num_ent, num_word = mention_word_tokens.shape

        # Reshape to two dimensions - needed because nn.Embedding only allows lookup with 2D-Tensors
        mention_gram_tokens = mention_gram_tokens.view(-1, num_gram)
        context_word_tokens = context_word_tokens.view(-1, num_context)
        mention_word_tokens = mention_word_tokens.view(-1, num_word)
        candidate_word_tokens = candidate_word_tokens.view(-1, num_word)
        candidate_gram_tokens = candidate_gram_tokens.view(-1, num_gram)
        candidate_ids = candidate_ids.view(-1, num_cand)

        # Get the embeddings
        mention_gram_embs = self.gram_embs(mention_gram_tokens)
        candidate_gram_embs = self.gram_embs(candidate_gram_tokens)
        mention_word_embs = self.mention_embs(mention_word_tokens)
        candidate_word_embs = self.ent_mention_embs(candidate_word_tokens)
        context_word_embs = self.word_embs(context_word_tokens)
        candidate_ent_embs = self.ent_embs(candidate_ids)

        # Reshape to original shape
        candidate_gram_embs = candidate_gram_embs.view(num_abst * num_ent, num_cand, num_gram, -1)
        candidate_word_embs = candidate_word_embs.view(num_abst * num_ent, num_cand, num_word, -1)

        # Sum the embeddings over the small and large tokens dimension
        mention_gram_embs = torch.mean(mention_gram_embs, dim=1)
        mention_word_embs = torch.mean(mention_word_embs, dim=1)
        candidate_gram_embs = torch.mean(candidate_gram_embs, dim=2)
        candidate_word_embs = torch.mean(candidate_word_embs, dim=2)

        if self.args.norm_gram:
            try:
                mention_gram_embs = F.normalize(mention_gram_embs, dim=1)
                candidate_gram_embs = F.normalize(candidate_gram_embs, dim=2)
            except RuntimeError:
                pass

        if self.args.norm_mention:
            try:
                mention_word_embs = F.normalize(mention_word_embs, dim=1)
                candidate_word_embs = F.normalize(candidate_word_embs, dim=2)
            except RuntimeError:
                pass

        context_word_embs = torch.mean(context_word_embs, dim=1)
        context_word_embs = self.orig_linear(context_word_embs)

        if self.args.norm_context:
            try:
                context_word_embs = F.normalize(context_word_embs, dim=1)
            except RuntimeError:
                pass

        # Concatenate word / gram embeddings
        combined_ent = torch.cat((candidate_ent_embs, candidate_gram_embs, candidate_word_embs), dim=2)
        combined_mention = torch.cat((context_word_embs, mention_gram_embs, mention_word_embs), dim=1)

        if self.args.norm_final:
            try:
                combined_ent = F.normalize(combined_ent, dim=2)
                combined_mention = F.normalize(combined_mention, dim=1)
            except RuntimeError:
                pass
            
        combined_mention.unsqueeze_(1)

        # Dot product over last dimension
        scores = (combined_mention * combined_ent).sum(dim=2)

        return scores