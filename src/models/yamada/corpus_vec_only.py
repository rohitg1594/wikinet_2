# Yamada model that also uses corpus vec
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.yamada.yamada_base import YamadaBase
from src.models.loss import Loss


class YamadaCorpusVecOnly(YamadaBase, Loss):

    def __init__(self, yamada_model=None, args=None):
        super().__init__(yamada_model, args)

        self.hidden = nn.Linear(6 + 3 * self.emb_dim, self.args.hidden_size)
        self.output = nn.Linear(self.args.hidden_size, 1)

    def forward(self, inputs):

        # Unpack / Get embs
        corpus_context, _, candidate_ids, _, _, _, _ = inputs
        b, num_doc, num_context = corpus_context.shape

        candidate_embs = self.ent_embs(candidate_ids)
        corpus_embs = self.word_embs(corpus_context)

        # Aggregate context
        corpus_embs = corpus_embs.mean(dim=2)
        corpus_embs = F.normalize(self.orig_linear(corpus_embs), dim=2)
        corpus_embs = corpus_embs.mean(dim=1)
        corpus_embs.unsqueeze_(1)

        # Get scores
        scores = (corpus_embs * candidate_embs).sum(dim=2)
        scores = scores.view(b, -1)

        return scores, corpus_embs, input

    def loss(self, scores, labels):
        return self.cross_entropy(scores, labels)
