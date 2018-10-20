# This module implements dataloader for the yamada model
import numpy as np
import torch
import torch.utils.data

from os.path import join

from src.utils.utils import reverse_dict, get_normalised_forms, equalize_len, normalise_form
from src.utils.data import pickle_load


class YamadaDataset(object):

    def __init__(self,
                 ent_prior=None,
                 ent_conditional=None,
                 yamada_model=None,
                 data=None,
                 args=None,
                 cand_rand=False,
                 cand_type='necounts'):
        super().__init__()

        self.args = args
        self.num_candidates = self.args.num_candidates
        self.cand_gen = self.num_candidates // 2
        self.ent2id = yamada_model['ent_dict']
        self.id2ent = reverse_dict(self.ent2id)
        self.word_dict = yamada_model['word_dict']
        self.max_ent = len(self.ent2id)
        self.ent_prior = ent_prior
        self.ent_conditional = ent_conditional

        self.cand_rand = cand_rand
        self.cand_type = cand_type
        if self.cand_rand:
            self.num_candidates = 10 ** 6
        if cand_type == 'necounts':
            # This is of the form: mention_str :  Counter(cand_id: counts)
            self.necounts = pickle_load(join(self.args.data_path, "necounts", "normal_necounts.pickle"))

        id2context, examples = data
        self.examples = examples
        self.id2context = id2context
        print(f'len examples: {len(examples)}, len id2contex: {len(id2context)}')
        self.processed_id2context = {}
        for index in self.id2context.keys():
            self.processed_id2context[index] = self._init_context(index)

        if self.args.model_name == 'corpus_vec':
            self.corpus_flag = True
            if self.args.num_docs > len(self.id2context):
                self.rand_docs = False
                self.corpus_context = np.vstack([context_arr for context_arr in self.processed_id2context.values()])
            else:
                self.rand_docs = True
        else:
            self.corpus_flag = False

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

        return complete_cands.astype(np.int64)

    def _init_context(self, index):
        """Initialize numpy array that will hold all context word tokens. Also return mentions"""

        context = self.id2context[index]
        if self.args.ignore_init:
            context = context[5:]
        if len(context) > 0:
            if isinstance(context[0], str):
                context = [self.word_dict.get(token, 0) for token in context]
        context = np.array(equalize_len(context, self.args.max_context_size))

        return context

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        # Initialize
        exact_match = np.zeros(self.num_candidates).astype(np.float32)
        contains = np.zeros(self.num_candidates).astype(np.float32)
        priors = np.zeros(self.num_candidates).astype(np.float32)
        conditionals = np.zeros(self.num_candidates).astype(np.float32)

        context_id, example = self.examples[index]
        context = self.processed_id2context[context_id]
        mention_str, ent_str, _, _ = example
        true_ent = self.ent2id.get(ent_str, 0)

        if self.corpus_flag:
            if self.rand_docs:
                other_docs = [self.processed_id2context[index]
                              for index in np.random.randint(0, high=len(self.processed_id2context),
                                                             size=self.args.num_docs - 1)]
                try:
                    corpus_context = np.vstack(list(self.processed_id2context[context_id]) + other_docs)
                except:
                    for i, doc in enumerate(other_docs):
                        print(i, doc.shape)
                    print(self.processed_id2context[context_id].shape)
            else:
                corpus_context = self.corpus_context

        nfs = get_normalised_forms(mention_str)
        candidate_ids = []
        for nf in nfs:
            if nf in self.necounts:
                candidate_ids.extend(self.necounts[nf])

        if true_ent in candidate_ids:
            candidate_ids.remove(true_ent)
        candidate_ids = self._gen_cands(true_ent, candidate_ids)

        for cand_idx, cand_id in enumerate(candidate_ids):
            ent_str = self.id2ent.get(cand_id, '')
            if mention_str == ent_str or mention_str in ent_str:
                exact_match[cand_idx] = 1
            if ent_str.startswith(mention_str) or ent_str.endswith(mention_str):
                contains[cand_idx] = 1

            priors[cand_idx] = self.ent_prior.get(cand_id, 0)
            nf = normalise_form(mention_str)
            if nf in self.ent_conditional:
                conditionals[cand_idx] = self.ent_conditional[nf].get(cand_id, 0)
            else:
                conditionals[cand_idx] = 0

        if self.corpus_flag:
            return corpus_context, context, candidate_ids, priors, conditionals, exact_match, contains
        else:
            return context, candidate_ids, priors, conditionals, exact_match, contains

    def __len__(self):
        return len(self.examples)

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
