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

        # Unpack / Get embs
        context, candidate_ids, _, _, _, _, corpus_context = inputs
        b, _ = context.shape
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

        # Combine context
        combined_context = torch.cat((context_embs, corpus_embs), dim=1)
        combined_context.unsqueeze_(1)

        # Get scores
        scores = (combined_context * candidate_embs).sum(dim=2)
        scores = scores.view(b, -1)

        return scores, corpus_embs, input

    def loss(self, scores, labels):
        return self.cross_entropy(scores, labels)
