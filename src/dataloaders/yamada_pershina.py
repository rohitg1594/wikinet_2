# This module implements dataloader for the yamada model with pershina candidates
import numpy as np
import torch
import torch.utils.data

from os.path import join

from src.utils.utils import reverse_dict, get_normalised_forms
from src.utils.data import pickle_load


class YamadaDataloader(object):

    def __init__(self,
                 ent_prior=None,
                 ent_conditional=None,
                 yamada_model=None,
                 data=None,
                 args=None,
                 cand_rand=False,
                 cand_type='pershina'):
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
        self.cand_type = cand_type

        if self.cand_rand:
            self.num_candidates = 10 ** 6

        # This is of the form: mention_str :  Counter(cand_id: counts)
        self.necounts = pickle_load(join(self.args.data_path, "necounts", "normal_necounts.pickle"))

    def _gen_cands(self, true_ent, candidates):

        if not self.cand_rand:
            if len(candidates) > self.cand_gen:
                cand_gen = np.random.choice(np.array(candidates), replace=False, size=self.cand_gen)
                cand_random = np.random.randint(0, self.max_ent, size=self.num_candidates - self.cand_gen - 1)
            else:
                cand_gen = np.array(candidates)
                cand_random = np.random.randint(0, self.max_ent, size=self.num_candidates - len(candidates) - 1)
            complete_cands = np.concatenate((np.array(true_ent)[None], cand_gen, cand_random))
        else:
            cand_random = np.random.randint(0, self.max_ent, size=self.num_candidates - 1)
            complete_cands = np.concatenate((np.array(true_ent)[None], cand_random))

        return complete_cands

    def _init_feats(self, num):
        arr = np.zeros((self.args.max_ent_size, self.num_candidates)).astype(np.float32)
        res = []
        for _ in range(num):
            res.append(arr)
        return tuple(res)

    def _update_string_feats(self, mention_str, candidates, ent_idx, exact_match, contains):
        for c_idx, candidate in enumerate(candidates[ent_idx]):
            ent_str = self.id2ent.get(candidate, '')
            if mention_str == ent_str or mention_str in ent_str:
                exact_match[ent_idx, c_idx] = 1
            if ent_str.startswith(mention_str) or ent_str.endswith(mention_str):
                contains[ent_idx, c_idx] = 1

    def _update_stat_feats(self, mention_str, candidates, ent_idx, priors, conditionals):
        for c_idx, candidate in enumerate(candidates[ent_idx]):
            priors[ent_idx, c_idx] = self.ent_prior.get(candidate, 0)
            if mention_str in self.ent_conditional:
                conditionals[ent_idx, c_idx] = self.ent_conditional[mention_str].get(candidate, 0)
            else:
                conditionals[ent_idx, c_idx] = 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        result = []

        # Each abstract is of shape num_ents * NUMBER_CANDIDATES
        all_candidates = np.zeros((self.args.max_ent_size, self.num_candidates)).astype(np.int64)

        if self.args.include_string:
            exact_match, contains = self._init_feats(2)
        if self.args.include_stats:
            priors, conditionals = self._init_feats(2)

        words_array = np.zeros(self.args.max_context_size, dtype=np.int64)
        words, examples = self.data[index]
        mask = np.zeros(self.args.max_ent_size, dtype=np.float32)
        mask[:len(examples)] = 1

        if len(words) > self.args.max_context_size:
            words_array[:self.args.max_context_size] = words[:self.args.max_context_size]
        else:
            words_array[:len(words)] = words

        for ent_idx, (mention_str, candidates) in enumerate(examples[:self.args.max_ent_size]):
            candidate_ids = [self.ent2id.get(candidate, 0) for candidate in candidates]
            true_ent = candidate_ids[0]
            candidate_ids = candidate_ids[1:]

            if self.cand_type == 'necounts':
                nfs = get_normalised_forms(mention_str)
                candidate_ids = []
                for nf in nfs:
                    if nf in self.necounts:
                        candidate_ids.extend(self.necounts[nf])

                if true_ent in candidate_ids:
                    candidate_ids.remove(true_ent)

            all_candidates[ent_idx] = self._gen_cands(true_ent, candidate_ids)

            if self.args.include_string:
                self._update_string_feats(mention_str, all_candidates, ent_idx, exact_match, contains)

            if self.args.include_stats:
                self._update_stat_feats(mention_str, all_candidates, ent_idx, priors, conditionals)

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