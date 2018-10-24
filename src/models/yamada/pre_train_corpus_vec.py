# Yamada model that also uses corpus vec
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.yamada.yamada_base import YamadaBase
from src.models.loss import Loss


class PreTrainCorpus(YamadaBase, Loss):

    def __init__(self, yamada_model=None, args=None):
        super().__init__(yamada_model, args)

        self.context_hidden = nn.Linear(self.ent_dim, 150)
        self.corpus_hidden = nn.Linear(self.ent_dim, 150)
        self.emb_ent_new = nn.Embedding(self.num_ent, self.ent_dim, padding_idx=0, sparse=True)

    def forward(self, inputs):

        # Unpack
        context, candidate_ids, _, _, _, _, corpus_context = inputs
        b, num_cand = candidate_ids.shape
        b, num_doc, num_context = corpus_context.shape
        b, num_context = context.shape

        # Get the embeddings
        candidate_embs = self.embeddings_ent(candidate_ids)
        context_embs = self.embeddings_word(context)
        corpus_embs = self.embeddings_word(corpus_context)

        # Aggregate corpus context
        corpus_embs = corpus_embs.mean(dim=2)
        corpus_embs = F.normalize(self.orig_linear(corpus_embs), dim=2)
        corpus_embs = self.corpus_hidden(corpus_embs.mean(dim=1))

        # Aggregate context
        context_embs = context_embs.mean(dim=1)
        context_embs = F.normalize(self.orig_linear(context_embs), dim=2)
        context_embs = self.context_hidden(context_embs)

        # Combined context
        combined_context = torch.cat((co))
        context_embs = F.normalize(self.orig_linear(context_embs), dim=1)
        corpus_embs = F.normalize(self.orig_linear(corpus_embs), dim=1)
        context_embs.unsqueeze_(1)
        corpus_embs.unsqueeze_(1)

        # Dot product over last dimension
        doc_dot_product = (context_embs * candidate_embs).sum(dim=2)
        corpus_dot_product = (corpus_embs * candidate_embs).sum(dim=2)

        # Unsqueeze in second dimension
        doc_dot_product = doc_dot_product.unsqueeze(dim=2)
        corpus_dot_product = corpus_dot_product.unsqueeze(dim=2)
        priors = priors.unsqueeze(dim=2)
        conditionals = conditionals.unsqueeze(dim=2)
        exact_match = exact_match.unsqueeze(dim=2)
        contains = contains.unsqueeze(dim=2)

        # Create input for mlp
        context_embs = context_embs.expand(-1, num_cand, -1)
        corpus_embs = corpus_embs.expand(-1, num_cand, -1)
        input = torch.cat((context_embs, doc_dot_product, corpus_embs, corpus_dot_product,
                           candidate_embs, priors, conditionals, exact_match, contains), dim=2)

        # Scores
        scores = self.output(F.relu(self.dropout(self.hidden(input))))
        scores = scores.view(b, -1)

        return scores, context_embs, input

    def loss(self, scores, labels):
        return self.cross_entropy(scores, labels)
