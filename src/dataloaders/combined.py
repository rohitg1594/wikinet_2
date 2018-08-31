# Dataloader, based on args.include_word, can include and exclude word information
from os.path import join
import pickle

import numpy as np
import torch
import torch.utils.data

from src.utils import reverse_dict, equalize_len, get_normalised_forms
from src.tokenization.regexp_tokenizer import RegexpTokenizer


class CombinedDataSet(object):

    def __init__(self,
                 gram_dict=None,
                 word_dict=None,
                 gram_tokenizer=None,
                 ent2id=None,
                 data=None,
                 args=None):
        """
        data_path: directory path of training data files
        num_shards: number of parts that training file is divided into
        """
        super().__init__()

        self.ent2id = ent2id
        self.len_ent = len(self.ent2id)
        self.id2ent = reverse_dict(self.ent2id)

        self.gram_dict = gram_dict
        self.word_dict = word_dict
        self.gram_tokenizer = gram_tokenizer
        self.word_tokenizer = RegexpTokenizer()
        self.args = args

        self.candidate_generation = self.args.num_candidates // 2
        self.data = data

        # Candidates
        if not self.args.cand_gen_rand:
            with open(join(self.args.data_path, 'necounts', 'normal_necounts.pickle'), 'rb') as f:
                self.necounts = pickle.load(f)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        # Context Word Tokens
        context_word_tokens, mentions = self.data[index]
        context_word_tokens = [self.word_dict[token] for token in context_word_tokens if token in self.word_dict]
        context_word_tokens = np.array(equalize_len(context_word_tokens, self.args.max_context_size))

        # TODO - maybe this is too expensive
        context_word_tokens_array = np.zeros((self.args.max_ent_size, self.args.max_context_size), dtype=np.int64)
        context_word_tokens_array[:len(mentions)] = context_word_tokens

        all_candidate_ids = np.zeros((self.args.max_ent_size, self.args.num_candidates)).astype(np.int64)
        all_candidate_grams = np.zeros((self.args.max_ent_size, self.args.num_candidates, self.args.max_gram_size)).astype(np.int64)
        all_mention_gram_tokens = np.zeros((self.args.max_ent_size, self.args.max_gram_size)).astype(np.int64)

        if self.args.include_word or self.args.include_mention:
            all_candidate_words = np.zeros(
                (self.args.max_ent_size, self.args.num_candidates, self.args.max_word_size)).astype(np.int64)
            all_mention_word_tokens = np.zeros((self.args.max_ent_size, self.args.max_word_size)).astype(np.int64)

        # Mask of indices to ignore for final loss
        mask = np.zeros(self.args.max_ent_size, dtype=np.float32)
        mask[:len(mentions)] = 1

        for ent_idx, (mention, ent_str) in enumerate(mentions[:self.args.max_ent_size]):
            if ent_str in self.ent2id:
                ent_id = self.ent2id[ent_str]
            else:
                continue

            # Mention Gram Tokens
            if self.args.include_gram:
                mention_gram_tokens = [self.gram_dict.get(token, 0) for token in self.gram_tokenizer(mention)]
                mention_gram_tokens = equalize_len(mention_gram_tokens, self.args.max_gram_size)
                all_mention_gram_tokens[ent_idx] = np.array(mention_gram_tokens, dtype=np.int64)

            # Mention Word Tokens
            if self.args.include_word:
                mention_word_tokens = [self.word_dict.get(token.lower(), 0)
                                       for token in self.word_tokenizer.tokenize(mention)]
                mention_word_tokens = equalize_len(mention_word_tokens, self.args.max_word_size)
                all_mention_word_tokens[ent_idx] = np.array(mention_word_tokens, dtype=np.int64)

            # Candidate Generation
            if self.args.cand_gen_rand:
                candidate_ids = np.concatenate((np.array(ent_id)[None],
                                                np.random.randint(1, self.len_ent + 1, size=self.args.num_candidates - 1)))
            else:
                nfs = get_normalised_forms(mention)
                candidates = []
                for nf in nfs:
                    if nf in self.necounts:
                        candidates.extend(self.necounts[nf])

                candidate_ids = [self.ent2id.get(candidate, 0) for candidate in candidates]

                if ent_id in candidate_ids:  # Remove if true entity is part of candidates
                    candidate_ids.remove(ent_id)

                if len(candidate_ids) > self.candidate_generation:
                    cand_generation = np.random.choice(np.array(candidate_ids), replace=False,
                                                       size=self.candidate_generation)
                    cand_random = np.random.randint(1, self.len_ent + 1,
                                                    self.args.num_candidates - self.candidate_generation - 1)
                else:
                    cand_generation = np.array(candidate_ids)
                    cand_random = np.random.randint(1, self.len_ent + 1, self.args.num_candidates - len(candidate_ids) - 1)

                candidate_ids = np.concatenate((np.array(ent_id)[None], cand_generation, cand_random))

            all_candidate_ids[ent_idx] = candidate_ids

            # Gram and word tokens for Candidates
            candidate_gram_tokens_matr = np.zeros((self.args.num_candidates, self.args.max_gram_size)).astype(np.int64)
            if self.args.include_word:
                candidate_word_tokens_matr = np.zeros((self.args.num_candidates, self.args.max_word_size)).astype(
                    np.int64)

            for cand_idx, candidate_id in enumerate(candidate_ids):
                candidate_str = self.id2ent.get(candidate_id, '').replace('_', ' ')

                # Candidate Gram Tokens
                if self.args.include_gram:
                    candidate_gram_tokens = [self.gram_dict[token] for token in self.gram_tokenizer(candidate_str)
                                             if token in self.gram_dict]
                    candidate_gram_tokens = equalize_len(candidate_gram_tokens, self.args.max_gram_size)
                    candidate_gram_tokens_matr[cand_idx] = np.array(candidate_gram_tokens, dtype=np.int64)

                # Candidate Word Tokens
                if self.args.include_word:
                    candidate_word_tokens = [self.word_dict.get(token.lower(), 0)
                                             for token in self.word_tokenizer.tokenize(candidate_str)]
                    candidate_word_tokens = equalize_len(candidate_word_tokens, self.args.max_word_size)
                    candidate_word_tokens_matr[cand_idx] = np.array(candidate_word_tokens, dtype=np.int64)

            all_candidate_grams[ent_idx] = candidate_gram_tokens_matr
            if self.args.include_word:
                all_candidate_words[ent_idx] = candidate_word_tokens_matr

        if self.args.only_prior:
            return (mask,
                    all_mention_word_tokens,
                    all_candidate_ids)

        elif self.args.include_word or self.args.include_mention:
            return (mask,
                    all_mention_gram_tokens,
                    all_mention_word_tokens,
                    context_word_tokens_array,
                    all_candidate_grams,
                    all_candidate_words,
                    all_candidate_ids)
        else:
            return (mask,
                    all_mention_gram_tokens,
                    context_word_tokens_array,
                    all_candidate_grams,
                    all_candidate_ids)

    def __len__(self):
        return len(self.data)

    def get_loader(self,
                   batch_size=1,
                   shuffle=False,
                   sampler=None,
                   pin_memory=True,
                   drop_last=False,
                   num_workers=4
                   ):

        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)
