# Model that only uses context and gram information
import torch
import torch.nn.functional as F

from src.models.combined.combined_base import CombinedBase


class CombinedContextGram(CombinedBase):

    def __init__(self, yamada_model=None, gram_embs=None, args=None):
        super().__init__(yamada_model, gram_embs, args)

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

        if self.args.norm_gram:
            mention_gram_embs = F.normalize(mention_gram_embs, dim=1)
            candidate_gram_embs = F.normalize(candidate_gram_embs, dim=2)

        context_word_embs = torch.mean(context_word_embs, dim=1)
        context_word_embs = self.orig_linear(context_word_embs)

        if self.args.norm_context:
            context_word_embs = F.normalize(context_word_embs, dim=1)

        # Concatenate word / gram embeddings
        combined_ent = torch.cat((candidate_ent_embs, candidate_gram_embs), dim=2)
        combined_mention = torch.cat((context_word_embs, mention_gram_embs), dim=1)

        if self.args.norm_final:
            combined_ent = F.normalize(combined_ent, dim=2)
            combined_mention = F.normalize(combined_mention, dim=1)
        combined_mention.unsqueeze_(1)

        # Dot product over last dimension
        scores = (combined_mention * combined_ent).sum(dim=2)

        return scores