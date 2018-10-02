# This model takes into account all information - context, word and grams
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase


class VAE(CombinedBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_size = self.args.max_word_size * self.args.mention_word_dim + self.args.max_gram_size*self.args.gram_dim
        print('IN SIZE : {}'.format(in_size))

        self.fc1 = nn.Linear(in_size, in_size // 2)
        self.fc2 = nn.Linear(in_size // 2, in_size // 4)
        self.fc31 = nn.Linear(in_size // 4, 20)
        self.fc32 = nn.Linear(in_size // 4, 20)
        self.fc4 = nn.Linear(20, in_size // 4)
        self.fc5 = nn.Linear(in_size // 4, in_size // 2)
        self.fc6 = nn.Linear(in_size // 2, in_size)

        # Unpack args
        mention_embs = kwargs['mention_embs']
        ent_mention_embs = kwargs['ent_mention_embs']
        gram_embs = kwargs['gram_embs']

        # Mention embeddings
        self.mention_embs = nn.Embedding(*mention_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.mention_embs.weight.data.copy_(mention_embs)

        # Entity mention embeddings
        self.ent_mention_embs = nn.Embedding(*ent_mention_embs.shape, padding_idx=0, sparse=self.args.sparse)
        self.ent_mention_embs.weight.data.copy_(ent_mention_embs)

        # Linear
        self.lin = nn.Linear(mention_embs.shape[1] + gram_embs.shape[1], mention_embs.shape[1] + gram_embs.shape[1],
                             bias=False)
        torch.nn.init.eye(self.lin.weight)

    def encode(self, x):
        h = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc31(h), self.fc32(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def forward(self, inputs):
        mention_gram_tokens, mention_word_tokens, context_word_tokens,  candidate_gram_tokens,  candidate_ids = inputs

        num_abst, num_ent, num_cand, num_gram = candidate_gram_tokens.shape
        num_abst, num_ent, num_word = mention_word_tokens.shape
        num_abst, num_ent, num_context = context_word_tokens.shape

        # Reshape to two dimensions - needed because nn.Embedding only allows lookup with 2D-Tensors
        mention_gram_tokens = mention_gram_tokens.view(-1, num_gram)
        mention_word_tokens = mention_word_tokens.view(-1, num_word)
        context_word_tokens = context_word_tokens.view(-1, num_context)
        candidate_gram_tokens = candidate_gram_tokens.view(-1, num_gram)
        candidate_ids = candidate_ids.view(-1, num_cand)

        # Get the embeddings
        mention_gram_embs = self.gram_embs(mention_gram_tokens)
        candidate_gram_embs = self.gram_embs(candidate_gram_tokens)
        mention_word_embs = self.word_embs(mention_word_tokens)
        context_word_embs = self.word_embs(context_word_tokens)
        candidate_ent_embs = self.ent_embs(candidate_ids)

        # Reshape to original shape
        candidate_gram_embs = candidate_gram_embs.view(num_abst * num_ent, num_cand, num_gram, -1)

        # Sum the embeddings over the small and large tokens dimension
        mention_gram_embs = torch.mean(mention_gram_embs, dim=1)
        candidate_gram_embs = torch.mean(candidate_gram_embs, dim=2)

        # if self.args.norm_gram:
        #     mention_gram_embs = F.normalize(mention_gram_embs, dim=1)
        #     candidate_gram_embs = F.normalize(candidate_gram_embs, dim=2)

        mention_word_embs = torch.mean(mention_word_embs, dim=1)
        mention_word_embs = self.orig_linear(mention_word_embs)

        candidate_word_embs = torch.mean(candidate_word_embs, dim=2)
        candidate_word_embs = self.orig_linear(candidate_word_embs)

        if self.args.norm_word:
            mention_word_embs = F.normalize(mention_word_embs, dim=1)
            candidate_word_embs = F.normalize(candidate_word_embs, dim=2)

        context_word_embs = torch.mean(context_word_embs, dim=1)
        context_word_embs = self.orig_linear(context_word_embs)

        if self.args.norm_context:
            context_word_embs = F.normalize(context_word_embs, dim=1)

        # Concatenate word / gram embeddings
        combined_ent = torch.cat((candidate_ent_embs, candidate_gram_embs, candidate_word_embs), dim=2)
        combined_mention = torch.cat((context_word_embs, mention_gram_embs, mention_word_embs), dim=1)

        if self.args.norm_final:
            combined_ent = F.normalize(combined_ent, dim=2)
            combined_mention = F.normalize(combined_mention, dim=1)
        combined_mention.unsqueeze_(1)

        # Dot product over last dimension
        scores = (combined_mention * combined_ent).sum(dim=2)

        return scores