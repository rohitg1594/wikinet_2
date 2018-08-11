# Yamada model that uses context, stat and string features.
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.yamada.yamada_base import YamadaBase


class YamadaContextStatsString(YamadaBase):

    def __init__(self, yamada_model=None, args=None):
        super().__init__(yamada_model, args)

        self.hidden = nn.Linear(5 + 2 * self.emb_dim, self.args.hidden_size)
        self.output = nn.Linear(self.args.hidden_size, 1)

    def forward(self, inputs):
        tokens, candidate_ids, priors, conditionals, exact_match, contains = inputs
        be, max_ent, num_cand = candidate_ids.shape
        candidate_ids = candidate_ids.view(-1, num_cand)

        # Get the embeddings
        candidate_embs = self.embeddings_ent(candidate_ids)
        candidate_embs = candidate_embs.view(be, max_ent, num_cand, -1)

        token_embs = self.embeddings_word(tokens)
        token_embs = token_embs.mean(dim=1)

        # Normalize / Pass through linear layer
        token_embs = F.normalize(self.orig_linear(token_embs), dim=1)
        out_vecs = token_embs
        token_embs.unsqueeze_(1)
        token_embs = token_embs.expand(be, max_ent, self.emb_dim)
        token_embs = token_embs.unsqueeze(2)

        # Dot product over last dimension / compute probabilites
        dot_product = (token_embs * candidate_embs).sum(dim=3)

        # Unsqueeze in third dimension
        dot_product.unsqueeze_(dim=3)
        priors.unsqueeze_(dim=3)
        conditionals.unsqueeze_(dim=3)
        exact_match.unsqueeze_(dim=3)
        contains.unsqueeze_(dim=3)

        # Create input for mlp
        token_embs = token_embs.expand(be, max_ent, num_cand, self.emb_dim)
        input = torch.cat((token_embs, dot_product, candidate_embs,
                           priors, conditionals, exact_match, contains), dim=3)

        scores = self.output(F.relu(self.dropout(self.hidden(input))))
        scores.squeeze_(dim=3)
        scores = scores.view(be * max_ent, -1)

        return scores, out_vecs
