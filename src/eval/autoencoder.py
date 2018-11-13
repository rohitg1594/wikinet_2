# Validator for string autoencoder model
import torch

from logging import getLogger
from os.path import join

from src.utils.utils import eval_ranking, equalize_len_w_eot, chunks, mse

import faiss
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import manifold

np.random.seed(1)

logger = getLogger(__name__)


class AutoencoderValidator:

    def __init__(self,
                 dev_strs=None,
                 num_clusters=None,
                 num_members=None,
                 char_dict=None,
                 rank_sample=None,
                 verbose=False,
                 ent_arr=None,
                 mention_arr=None,
                 args=None,
                 dev_arr=None,
                 gold=None,
                 model_dir=None):

        self.dev_strs = dev_strs
        self.dev_arr = dev_arr
        self.num_clusters = num_clusters
        self.num_members = num_members
        self.char_dict = char_dict
        self.rank_sample = rank_sample
        self.verbose = verbose
        self.args = args
        self.max_char_size = self.args.max_char_size
        self.model_dir = model_dir

        self.valid_mask = np.random.choice(len(self.dev_strs), self.rank_sample)
        self.valid_strs = [self.dev_strs[i] for i in self.valid_mask]

        self.tsne_strs, self.tsne_arr = self.create_tsne_strs()
        self.X_init = 'pca'

        self.ent_arr = ent_arr
        self.mention_arr = mention_arr
        if self.args.use_cuda:
            self.ent_arr = self.ent_arr.cuda(self.args.device)
            self.mention_arr = self.mention_arr.cuda(self.args.device)
            self.tsne_arr = (torch.from_numpy(self.tsne_arr)).cuda(self.args.device)

        # Gold
        self.gold = gold

        self.valid_metrics = []

    def create_tsne_strs(self):

        logger.info('sorting strs.....')
        sorted_strs = sorted(self.dev_strs)
        logger.info('strs sorted.')

        # Create list of tsne mentions string
        tsne_strs = []
        starts = np.random.choice(np.arange(len(sorted_strs)), self.num_clusters)
        for start in starts:
            tsne_strs.extend(sorted_strs[start:start + 10])

        # Create tsne char id array
        tsne_arr = np.zeros((len(tsne_strs), self.max_char_size), dtype=np.int64)
        for i, tsne_str in enumerate(tsne_strs):
            char_ids = [self.char_dict[char] for char in list(tsne_str)]
            tsne_arr[i] = equalize_len_w_eot(char_ids, self.args.max_char_size, self.char_dict['EOT'])

        return tsne_strs, tsne_arr

    def plot_embedding(self, X, epoch):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure(figsize=(12, 12))
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], self.tsne_strs[i], fontdict={'weight': 'bold', 'size': 9})

        plt.savefig(join(self.model_dir, f'tsne-{epoch}.png'))

    def plot_tsne(self, model, epoch):
        model.eval()
        _, X, _ = model(self.tsne_arr)

        logger.info("Computing t-SNE embedding.....")
        tsne = manifold.TSNE(n_components=2, init=self.X_init, random_state=0, method='exact', learning_rate=100)
        logger.info("Done.")

        X_tsne = tsne.fit_transform(X.detach().cpu().numpy())
        self.X_init = X_tsne

        self.plot_embedding(X_tsne, epoch)

    @staticmethod
    def id_to_str(strs, ranks):

        s = ''
        for i, rank in enumerate(ranks):
            s += strs[i] + '|'
            for id in rank:
                s += strs[id] + ','

            s += '\n'

        return s

    def validate(self, model, plot_tsne=True, epoch=None):
        model.eval()
        full_loss = 0
        all_hidden = np.zeros((len(self.dev_strs), self.args.hidden_size))
        bs = self.args.batch_size

        for batch_idx, batch in enumerate(chunks(self.dev_arr, bs)):
            batch = torch.from_numpy(batch)
            if self.args.use_cuda:
                batch = batch.cuda(self.args.device)
            input, hidden, output = model(batch)
            all_hidden[batch_idx * bs:(batch_idx + 1) * bs] = hidden.detach()
            loss = mse(input, output)
            full_loss += loss.item()

        if plot_tsne: self.plot_tsne(model, epoch)

        _, ent_encoded, _ = model(self.ent_arr)
        ent_encoded = ent_encoded.detach().cpu().numpy()
        _, mentions_encoded, _ = model(self.mention_arr)
        mentions_encoded = mentions_encoded.detach().cpu().numpy()

        index = faiss.IndexFlatL2(ent_encoded.shape[1])
        index.add(ent_encoded)

        _, predictions = index.search(mentions_encoded, 100)
        print(f'predictions : {predictions.shape}, gold: {len(self.gold)}')
        results = eval_ranking(predictions, self.gold, [1, 10, 100])
        self.valid_metrics.append(results[0])

        return full_loss / (batch_idx + 1), results
