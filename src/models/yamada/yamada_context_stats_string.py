# Yamada model that uses context, stat and string features.
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.yamada.yamada_base import YamadaBase
from src.models.loss import Loss


class YamadaContextStatsString(YamadaBase, Loss):

    def __init__(self, yamada_model=None, args=None):
        super().__init__(yamada_model, args)

        self.hidden = nn.Linear(5 + 2 * self.emb_dim, self.args.hidden_size)
        self.output = nn.Linear(self.args.hidden_size, 1)

        self.dp = nn.Dropout(args.dp)

    def forward(self, input_dict):

        # Unpack
        candidate_ids = input_dict['cand_ids']
        context = input_dict['context']
        priors = input_dict['priors']
        conditionals = input_dict['conditionals']
        exact_match = input_dict['exact_match']
        contains = input_dict['contains']
        b, num_cand = candidate_ids.shape

        print(f'CANDIDATE IDS SHAPE : {candidate_ids.shape}, SAMPLE \n: {candidate_ids[:5, :5]}')
        print(f'CONTEXT SHAPE : {context.shape}, SAMPLE \n: {context[:5, :5]}')
        print(f'PRIORS SHAPE : {priors.shape}, SAMPLE \n: {priors[:5, :5]}')
        print(f'CONDITIONALS SHAPE : {conditionals.shape}, SAMPLE \n: {conditionals[:5, :5]}')
        print(f'EXACT MATCH SHAPE : {exact_match.shape}, SAMPLE \n: {exact_match[:5, :5]}')
        print(f'CONTAINS SHAPE : {contains.shape}, SAMPLE \n: {contains[:5, :5]}')
        sys.exit(0)

        # Get the embeddings
        candidate_embs = self.dp(self.ent_embs(candidate_ids))
        context_embs = self.dp(self.word_embs(context))

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

