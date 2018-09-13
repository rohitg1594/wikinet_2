# This module implements dataloader for the yamada model with necounts candidates
import numpy as np
import torch
import torch.utils.data

from os.path import join

from src.utils.utils import get_normalised_forms, reverse_dict
from src.utils.data import pickle_load


class YamadaConllDataset(object):

    def __init__(self,
                 args=None,
                 data=None,
                 ent_prior=None,
                 ent_conditional=None,
                 yamada_model=None):
        super().__init__()

        self.args = args
        self.num_candidates = self.args.num_candidates
        self.ent2id = yamada_model['ent_dict']
        self.id2ent = reverse_dict(self.ent2id)
        self.cand_generation = self.num_candidates // 2
        self.ent_prior = ent_prior
        self.ent_conditional = ent_conditional
        self.data = data

        self.necounts = pickle_load(join(self.args.data_path, "normal_necounts.pickle"))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        result = []

        # Each abstract is of shape num_ents * NUMBER_CANDIDATES
        all_candidates = np.zeros((self.args.max_ent_size, self.num_candidates)).astype(np.int64)
        labels = np.zeros(self.args.max_ent_size).astype(np.int64)

        if self.args.include_string:
            exact_match = np.zeros((self.args.max_ent_size, self.num_candidates)).astype(np.int64)
            contains = np.zeros((self.args.max_ent_size, self.num_candidates)).astype(np.int64)
        if self.args.include_stats:
            candidate_priors = np.zeros((self.args.max_ent_size, self.num_candidates))
            candidate_conditionals = np.zeros((self.args.max_ent_size, self.num_candidates))

        words_array = np.zeros(self.args.max_context_size, dtype=np.int64)
        words, values = self.data[index]
        mask = np.zeros(self.args.max_ent_size, dtype=np.float32)
        mask[:len(values)] = 1

        if len(words) > self.args.max_context_size:
            words_array[:self.args.max_context_size] = words[:self.args.max_context_size]
        else:
            words_array[:len(words)] = words

        for ent_idx, (mention_str, true_ent) in enumerate(values[:self.args.max_ent_size]):
            nfs = get_normalised_forms(mention_str)
            candidates = []
            for nf in nfs:
                if nf in self.necounts:
                    candidates.extend(self.necounts[nf])

            candidates_id = [self.ent2id.get(candidate, 0) for candidate in candidates]
            # for candidate_id in candidates_id:
            #   candidate_conditionals[ent_idx, ]

            if true_ent in candidates_id:
                candidates_id.remove(true_ent)

            if len(candidates_id) > self.cand_generation:
                cand_generation = np.random.choice(np.array(candidates_id),
                                                   replace=False, size=self.cand_generation)
                cand_random = np.random.randint(0, len(self.ent2id),
                                                size=self.num_candidates - self.cand_generation - 1)
            else:
                cand_generation = np.array(candidates_id)
                cand_random = np.random.randint(0, len(self.ent2id),
                                                size=self.num_candidates - len(candidates_id) - 1)

            before = np.concatenate((np.array(true_ent)[None], cand_generation, cand_random))
            true_index = np.random.randint(len(before))
            labels[ent_idx] = true_index
            all_candidates[ent_idx] = np.roll(before, true_index)

            for c_idx, candidate in enumerate(all_candidates[ent_idx]):
                if self.args.include_stats:
                    candidate_priors[ent_idx, c_idx] = self.ent_prior.get(candidate, 0)
                    candidate_conditionals[ent_idx, c_idx] = self.ent_conditional[mention_str].get(candidate, 0)
                if self.args.include_string:
                    ent_str = self.id2ent.get(candidate, '')
                    if mention_str == ent_str or mention_str in ent_str:
                        exact_match[ent_idx, c_idx] = 1
                    if ent_str.startswith(mention_str) or ent_str.endswith(mention_str):
                        contains[ent_idx, c_idx] = 1

        result.extend([mask, labels, words_array, all_candidates])
        if self.args.include_stats:
            result.extend([candidate_priors, candidate_conditionals])
        if self.args.include_string:
            result.extend([exact_match, contains])

    def __len__(self):
        return len(self.data)

    def get_loader(self,
                   batch_size=1,
                   shuffle=False,
                   sampler=None,
                   pin_memory=True,
                   drop_last=True,
                   num_workers=4
                   ):

        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)
