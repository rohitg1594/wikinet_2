# Model that only tries to learn the prior probability through mention words, it predicts on all candidates
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase


class Full(CombinedBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unpack args
        mention_embs = kwargs['mention_embs']
        ent_mention_embs = kwargs['ent_mention_embs']
        args = kwargs['args']

        # Mention embeddings
        self.mention_embs = nn.Embedding(mention_embs.shape[0], mention_embs.shape[1], padding_idx=0,
                                         sparse=self.args.sparse)
        self.mention_embs.weight.data.copy_(mention_embs)
        self.mention_embs.weight.requires_grad = self.args.train_mention

        # Entity mention embeddings
        self.ent_mention_embs = nn.Embedding(ent_mention_embs.shape[0], ent_mention_embs.shape[1], padding_idx=0,
                                             sparse=self.args.sparse)
        self.ent_mention_embs.weight.data.copy_(ent_mention_embs)
        self.ent_mention_embs.weight.requires_grad = self.args.train_mention

        self.candidate_ids = torch.arange(1, len(self.ent_mention_embs.weight)).long()
        if args.use_cuda:
            if isinstance(self.args.device, int):
                self.candidate_ids = self.candidate_ids.cuda(self.args.device)

    def forward(self, inputs):
        mention_word_tokens = inputs[0]

        num_abst, num_ent, num_word = mention_word_tokens.shape

        # Reshape to two dimensions - needed because nn.Embedding only allows lookup with 2D-Tensors
        mention_word_tokens = mention_word_tokens.view(-1, num_word)

        # Get the embeddings
        mention_embs = self.mention_embs(mention_word_tokens)
        candidate_embs = self.ent_mention_embs(self.candidate_ids)

        # Sum the embeddings over the small and large tokens dimension
        mention_embs_agg = torch.mean(mention_embs, dim=1)

        # Normalize
        if self.args.norm_final:
            candidate_embs = F.normalize(candidate_embs, dim=1)
            mention_embs_agg = F.normalize(mention_embs_agg, dim=1)

        # Dot product over last dimension
        mention_embs_agg.unsqueeze_(1)
        candidate_embs.unsqueeze_(0)
        scores = (mention_embs_agg * candidate_embs).sum(dim=2)

        return scores
