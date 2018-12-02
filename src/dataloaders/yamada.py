# This module implements dataloader for the yamada model

import torch.utils.data
from src.utils.utils import *
from more_itertools import unique_everseen


class YamadaDataset(object):

    def __init__(self,
                 ent_prior=None,
                 ent_conditional=None,
                 yamada_model=None,
                 data=None,
                 args=None,
                 cand_rand=False,
                 cand_type='necounts',
                 split=None,
                 necounts=None,
                 redirects=None,
                 dis_dict=None):
        super().__init__()

        self.args = args
        self.num_candidates = self.args.num_candidates
        self.num_cand_gen = self.num_candidates // 2
        self.ent2id = yamada_model['ent_dict']
        self.len_ent = len(self.ent2id)
        self.id2ent = reverse_dict(self.ent2id)
        self.word_dict = yamada_model['word_dict']
        self.max_ent = len(self.ent2id)
        self.ent_prior = ent_prior
        self.ent_conditional = ent_conditional
        self.ent_strs = list(self.ent_prior.keys())
        self.split = split

        self.redirects = redirects
        self.dis_dict = dis_dict

        self.cand_rand = cand_rand
        self.cand_type = cand_type
        if self.cand_rand:
            self.num_candidates = 10 ** 6

        self.necounts = necounts

        id2context, examples = data
        self.examples = examples
        self.id2context = id2context

        processed_f_name = join(self.args.data_path, 'cache', f'processed_id2context_{self.args.data_type}_{self.split}')
        if os.path.exists(processed_f_name):
            self.processed_id2context = pickle_load(processed_f_name)
        else:
            self.processed_id2context = {}
            for index in self.id2context.keys():
                self.processed_id2context[index] = self._init_context(index)
            pickle_dump(self.processed_id2context, processed_f_name)

        if 'corpus_vec' in self.args.model_name:
            self.corpus_flag = True
            if self.args.num_docs > len(self.id2context):
                self.rand_docs = False
                self.corpus_context = np.vstack([context_arr for context_arr in self.processed_id2context.values()])
            else:
                self.rand_docs = True
        else:
            self.corpus_flag = False

    def add_dismb_cands(self, cands, mention):
        mention_title = mention.title().replace(' ', '_')
        cands.append(mention_title)
        if mention_title != self.redirects.get(mention_title, mention_title):
            cands.append(self.redirects[mention_title])
        mention_disamb = mention_title + '_(disambiguation)'

        if mention_title in self.dis_dict:
            cands.extend(self.dis_dict[mention_title])
        if mention_disamb in self.dis_dict:
            cands.extend(self.dis_dict[mention_disamb])

        return cands

    def _gen_cands(self, ent_str, mention):
        cand_gen_strs = self.add_dismb_cands([], mention)
        nfs = get_normalised_forms(mention)
        for nf in nfs:
            if nf in self.necounts:
                cand_gen_strs.extend(self.necounts[nf])

        cand_gen_strs = list(unique_everseen(cand_gen_strs[:self.num_cand_gen]))
        if ent_str in cand_gen_strs:
            not_in_cand = False
            cand_gen_strs.remove(ent_str)
        else:
            not_in_cand = True

        cand_strs = [ent_str] + cand_gen_strs + random.sample(self.ent_strs, self.args.num_candidates - len(cand_gen_strs) - 1)
        cand_ids = np.array([self.ent2id.get(cand_str, 0) for cand_str in cand_strs], dtype=np.int64)

        return cand_ids, cand_strs, not_in_cand

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

    def _gen_features(self, mention_str, cand_strs):

        # Initialize
        exact = np.zeros(self.num_candidates).astype(np.float32)
        contains = np.zeros(self.num_candidates).astype(np.float32)
        priors = np.zeros(self.num_candidates).astype(np.float32)
        conditionals = np.zeros(self.num_candidates).astype(np.float32)
        # print(f'MENTION STR - {mention_str}, CAND STRS - {cand_strs}')

        # Populate
        for cand_idx, cand_str in enumerate(cand_strs):
            if mention_str == cand_str or mention_str in cand_str:
                exact[cand_idx] = 1
            if cand_str.startswith(mention_str) or cand_str.endswith(mention_str):
                contains[cand_idx] = 1

            priors[cand_idx] = self.ent_prior.get(cand_str, 0) * 10**3  # priors are really small, rescale
            nf = normalise_form(mention_str)
            if nf in self.ent_conditional:
                conditionals[cand_idx] = self.ent_conditional[nf].get(cand_str, 0)
            else:
                conditionals[cand_idx] = 0

        return {'exact_match': exact,
                'contains': contains,
                'priors': priors,
                'conditionals': conditionals}

    def _get_corpus_context(self, context_id):
        if self.rand_docs:
            other_docs = [self.processed_id2context[index]
                          for index in np.random.randint(0, high=len(self.processed_id2context),
                                                         size=self.args.num_docs - 1)]
            full_corpus = list(self.processed_id2context[context_id][None, :]) + other_docs
            corpus_context = np.vstack(full_corpus)
        else:
            corpus_context = self.corpus_context

        return corpus_context

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        context_id, example = self.examples[index]
        context = self.processed_id2context[context_id]
        mention_str, ent_str, _, _ = example
        ent_str = self.redirects.get(ent_str, ent_str)
        cand_ids, cand_strs, not_in_cand = self._gen_cands(ent_str, mention_str)
        features_dict = self._gen_features(mention_str, cand_strs)

        output = {'cand_ids': cand_ids,
                  'not_in_cand': not_in_cand,
                  'context': context,
                  'cand_strs': cand_strs,
                  'ent_strs': ent_str,
                  **features_dict}

        if self.corpus_flag:
            corpus_context = self._get_corpus_context(context_id)
            output['corpus_context'] = corpus_context

        return output

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
