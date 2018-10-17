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

        # Unpack
        context, candidate_ids, priors, conditionals, exact_match, contains = inputs
        b, num_cand = candidate_ids.shape

        # Get the embeddings
        candidate_embs = self.embeddings_ent(candidate_ids)
        context_embs = self.embeddings_word(context)

        # Aggregate context
        context_embs = context_embs.mean(dim=1)

        # Normalize / Pass through linear layer / Unsqueeze
        context_embs = F.normalize(self.orig_linear(context_embs), dim=1)
        context_embs.unsqueeze_(1)

        # Dot product over last dimension
        dot_product = (context_embs * candidate_embs).sum(dim=2)

        # Unsqueeze in second dimension
        dot_product = dot_product.unsqueeze(dim=2)
        priors = priors.unsqueeze(dim=2)
        conditionals = conditionals.unsqueeze(dim=2)
        exact_match = exact_match.unsqueeze(dim=2)
        contains = contains.unsqueeze(dim=2)

        # Create input for mlp
        context_embs = context_embs.expand(-1, num_cand, -1)
        input = torch.cat((context_embs, dot_product, candidate_embs, priors, conditionals, exact_match, contains), dim=2)

        # Scores
        scores = self.output(F.relu(self.dropout(self.hidden(input))))
        scores = scores.view(b, -1)

        return scores, context_embs, input

    def loss(self, scores, labels):
        return self.cross_entropy(scores, labels)

