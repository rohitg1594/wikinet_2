# This model implements conditional variational autoencoder, conditioned on aggregated context embeddings
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.combined.base import CombinedBase


class VAE(CombinedBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_size = self.args.mention_word_dim + self.args.max_gram_size + 300
        print('IN SIZE : {}'.format(in_size))

        self.fc1 = nn.Linear(in_size, in_size // 2)
        self.fc2 = nn.Linear(in_size // 2, in_size // 4)
        self.fc31 = nn.Linear(in_size // 4, 20)
        self.fc32 = nn.Linear(in_size // 4, 20)
        self.fc4 = nn.Linear(20 + 300, 300)

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
        mention_gram_tokens, mention_word_tokens, context_word_tokens = inputs

        num_abst, num_ent, num_gram = mention_gram_tokens.shape
        num_abst, num_ent, num_word = mention_word_tokens.shape
        num_abst, num_ent, num_context = context_word_tokens.shape

        # Reshape to two dimensions - needed because nn.Embedding only allows lookup with 2D-Tensors
        mention_gram_tokens = mention_gram_tokens.view(-1, num_gram)
        mention_word_tokens = mention_word_tokens.view(-1, num_word)
        context_word_tokens = context_word_tokens.view(-1, num_context)

        # Get the embeddings
        mention_gram_embs = self.gram_embs(mention_gram_tokens)
        mention_word_embs = self.word_embs(mention_word_tokens)
        context_word_embs = self.word_embs(context_word_tokens)

        # Sum the embeddings over the small and large tokens dimension
        mention_gram_agg = torch.mean(mention_gram_embs, dim=1)
        mention_word_agg = torch.mean(mention_word_embs, dim=1)
        context_word_agg = self.orig_linear(torch.mean(context_word_embs, dim=1))

        # Pass through autoencoder
        encoder_input = torch.cat((mention_gram_agg, mention_word_agg, context_word_agg), dim=1)
        mu, logvar = self.encode(encoder_input)
        z = self.reparameterize(mu, logvar)
        ent_embs = self.decode(torch.cat([z, context_word_agg], dim=1))

        return ent_embs
