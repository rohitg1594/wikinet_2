# This module implements dataloader for the yamada model
import numpy as np
import torch
import torch.utils.data

from src.utils import reverse_dict


class YamadaPershina(object):

    def __init__(self,
                 ent_prior=None,
                 ent_conditional=None,
                 yamada_model=None,
                 data=None,
                 args=None):
        super().__init__()

        self.args = args
        self.number_candidates = self.args.num_candidates
        self.cand_generation = self.number_candidates // 2
        self.ent2id = yamada_model['ent_dict']
        self.id2ent = reverse_dict(self.ent2id)
        self.data = data
        self.max_ent = len(self.ent2id)
        self.ent_prior = ent_prior
        self.ent_conditional = ent_conditional

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        result = []

        # Each abstract is of shape num_ents * NUMBER_CANDIDATES
        all_candidates = np.zeros((self.args.max_ent_size, self.number_candidates)).astype(np.int64)
        labels = np.zeros(self.args.max_ent_size).astype(np.int64)

        if self.args.string_features:
            exact_match = np.zeros((self.args.max_ent_size, self.number_candidates)).astype(np.int64)
            contains = np.zeros((self.args.max_ent_size, self.number_candidates)).astype(np.int64)
        if self.args.stat_features:
            candidate_priors = np.zeros((self.args.max_ent_size, self.number_candidates)).astype(np.float32)
            candidate_conditionals = np.zeros((self.args.max_ent_size, self.number_candidates)).astype(np.float32)

        words_array = np.zeros(self.args.max_word_size, dtype=np.int64)
        words, values = self.data[index]
        mask = np.zeros(self.args.max_ent_size, dtype=np.float32)
        mask[:len(values)] = 1

        if len(words) > self.args.max_word_size:
            words_array[:self.args.max_word_size] = words[:self.args.max_word_size]
        else:
            words_array[:len(words)] = words

        for ent_idx, (mention_str, candidates) in enumerate(values[:self.args.max_ent_size]):
            candidates_id = [self.ent2id.get(candidate, 0) for candidate in candidates]
            true_ent = candidates_id[0]
            other_cands = candidates_id[1:]

            if len(other_cands) > self.cand_generation:
                cand_generation = np.random.choice(np.array(other_cands),
                                                   replace=False, size=self.cand_generation)
                cand_random = np.random.randint(0, self.max_ent,
                                                size=self.number_candidates - self.cand_generation - 1)
            else:
                cand_generation = np.array(other_cands)
                cand_random = np.random.randint(0, self.max_ent,
                                                size=self.number_candidates - len(other_cands) - 1)

            before = np.concatenate((np.array(true_ent)[None], cand_generation, cand_random))
            true_index = np.random.randint(len(before))
            labels[ent_idx] = true_index
            all_candidates[ent_idx] = np.roll(before, true_index)

            if self.args.string_features:
                for c_idx, candidate in enumerate(all_candidates[ent_idx]):
                    ent_str = self.id2ent.get(candidate, '')
                    if mention_str == ent_str or mention_str in ent_str:
                        exact_match[ent_idx, c_idx] = 1
                    if ent_str.startswith(mention_str) or ent_str.endswith(mention_str):
                        contains[ent_idx, c_idx] = 1

            if self.args.stat_features:
                for c_idx, candidate in enumerate(all_candidates[ent_idx]):
                    candidate_priors[ent_idx, c_idx] = self.ent_prior.get(candidate, 0)
                    if mention_str in self.ent_conditional:
                        candidate_conditionals[ent_idx, c_idx] = self.ent_conditional[mention_str].get(candidate, 0)
                    else:
                        candidate_conditionals[ent_idx, c_idx] = 0

        result.extend([mask, labels, words_array, all_candidates])
        if self.args.stat_features:
            result.extend([candidate_priors, candidate_conditionals])
        if self.args.string_features:
            result.extend([exact_match, contains])

        return result

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