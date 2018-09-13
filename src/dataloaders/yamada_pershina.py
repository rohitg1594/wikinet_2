# This module implements dataloader for the yamada model with pershina candidates
import numpy as np
import torch
import torch.utils.data

from src.utils.utils import reverse_dict


class YamadaPershina(object):

    def __init__(self,
                 ent_prior=None,
                 ent_conditional=None,
                 yamada_model=None,
                 data=None,
                 args=None,
                 cand_rand=False):
        super().__init__()

        self.args = args
        self.num_candidates = self.args.num_candidates
        self.cand_gen = self.num_candidates // 2
        self.ent2id = yamada_model['ent_dict']
        self.id2ent = reverse_dict(self.ent2id)
        self.data = data
        self.max_ent = len(self.ent2id)
        self.ent_prior = ent_prior
        self.ent_conditional = ent_conditional
        self.cand_rand = cand_rand

        if self.cand_rand:
            self.num_candidates = 10 ** 6

    def _gen_cands(self, true_ent, perhsina_cands):
        if not self.cand_rand:
            if len(perhsina_cands) > self.cand_gen:
                cand_gen = np.random.choice(np.array(perhsina_cands), replace=False, size=self.cand_gen)
                cand_random = np.random.randint(0, self.max_ent, size=self.num_candidates - self.cand_gen - 1)
            else:
                cand_gen = np.array(perhsina_cands)
                cand_random = np.random.randint(0, self.max_ent, size=self.num_candidates - len(perhsina_cands) - 1)
            cands = np.concatenate((np.array(true_ent)[None], cand_gen, cand_random))
        else:
            cand_random = np.random.randint(0, self.max_ent, size=self.num_candidates - 1)
            cands = np.concatenate((np.array(true_ent)[None], cand_random))

        return cands

    def _init_features(self, num):
        arr = np.zeros((self.args.max_ent_size, self.num_candidates)).astype(np.float32)
        res = []
        for _ in range(num):
            res.append(arr)
        return tuple(res)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        result = []

        # Each abstract is of shape num_ents * NUMBER_CANDIDATES
        all_candidates = np.zeros((self.args.max_ent_size, self.num_candidates)).astype(np.int64)
        labels = np.zeros(self.args.max_ent_size).astype(np.int64)

        if self.args.include_string:
            exact_match, contains = self._init_features(2)
        if self.args.include_stats:
            priors, conditionals = self._init_features(2)

        words_array = np.zeros(self.args.max_context_size, dtype=np.int64)
        words, values = self.data[index]
        mask = np.zeros(self.args.max_ent_size, dtype=np.float32)
        mask[:len(values)] = 1

        if len(words) > self.args.max_context_size:
            words_array[:self.args.max_context_size] = words[:self.args.max_context_size]
        else:
            words_array[:len(words)] = words

        for ent_idx, (mention_str, candidates) in enumerate(values[:self.args.max_ent_size]):
            candidates_id = [self.ent2id.get(candidate, 0) for candidate in candidates]
            true_ent = candidates_id[0]
            pershina_cands = candidates_id[1:]

            all_candidates[ent_idx] = self._gen_cands(true_ent, pershina_cands)

            #true_index = np.random.randint(len(before))
            #labels[ent_idx] = true_index
            #all_candidates[ent_idx] = np.roll(before, true_index)

            if self.args.include_string:
                for c_idx, candidate in enumerate(all_candidates[ent_idx]):
                    ent_str = self.id2ent.get(candidate, '')
                    if mention_str == ent_str or mention_str in ent_str:
                        exact_match[ent_idx, c_idx] = 1
                    if ent_str.startswith(mention_str) or ent_str.endswith(mention_str):
                        contains[ent_idx, c_idx] = 1

            if self.args.include_stats:
                for c_idx, candidate in enumerate(all_candidates[ent_idx]):
                    priors[ent_idx, c_idx] = self.ent_prior.get(candidate, 0)
                    if mention_str in self.ent_conditional:
                        conditionals[ent_idx, c_idx] = self.ent_conditional[mention_str].get(candidate, 0)
                    else:
                        conditionals[ent_idx, c_idx] = 0

        result.extend([mask, words_array, all_candidates])
        if self.args.include_stats:
            result.extend([priors, conditionals])
        if self.args.include_string:
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