# Validator class
import numpy as np
import faiss

from os.path import join

import re

from logging import getLogger

from src.utils import reverse_dict, equalize_len, normalize
from src.evaluation.eval_utils import eval_ranking
from src.conll.loaders import is_dev_doc, is_test_doc, is_training_doc, iter_docs

logger = getLogger()

RE_DOCID = re.compile('^\d+')


class Validator:
    def __init__(self,
                 gram_tokenizer=None,
                 gram_dict=None,
                 yamada_model=None,
                 data=None,
                 args=None,):

        self.tokenizer = gram_tokenizer
        self.word_dict = yamada_model['word_dict']
        self.rev_word_dict = reverse_dict(self.word_dict)
        self.ent_dict = yamada_model['ent_dict']
        self.rev_ent_dict = reverse_dict(self.ent_dict)
        self.gram_dict = gram_dict
        self.rev_gram_dict = reverse_dict(self.gram_dict)
        self.data = data
        self.args = args

        self.ent_gram_indices, self.ent_word_indices = self._gen_ent_tokens()
        (self.all_gold,
         self.mention_gram_indices_l,
         self.mention_word_indices_l,
         self.context_word_indices_l) = self._gen_wiki_mention_tokens()

        self.mention_gram_indices = np.vstack(self.mention_gram_indices_l).astype(np.int32)
        self.mention_word_indices = np.vstack(self.mention_word_indices_l).astype(np.int32)
        self.context_word_indices = np.vstack(self.context_word_indices_l).astype(np.int32)
        self.all_gold = np.array(self.all_gold).astype(np.int32)
        self.mask = np.random.choice(np.arange(len(self.mention_gram_indices)), size=self.args.query_size)

        if self.args.debug:
            for i in np.random.choice(range(len(self.all_gold)), size=10):
                s = ''
                m_g = self.mention_gram_indices[i]
                s += ''.join([self.rev_gram_dict[token][0] for token in m_g if token in self.rev_gram_dict]) + '|'
                m_w = self.mention_word_indices[i]
                s += ' '.join([self.rev_word_dict[token] for token in m_w if token in self.rev_word_dict]) + '|'
                c_w = self.context_word_indices[i][:20]
                s += ' '.join([self.rev_word_dict[token] for token in c_w if token in self.rev_word_dict]) + '|'
                s += self.rev_ent_dict[self.all_gold[i]]
                print(s)
                print()

    def _gen_ent_tokens(self):
        ent_gram_tokens = np.zeros((len(self.ent_dict) + 1, self.args.max_gram_size)).astype(np.int32)
        ent_word_tokens = np.zeros((len(self.ent_dict) + 1, self.args.max_word_size)).astype(np.int32)
        for ent_str, ent_id in self.ent_dict.items():
            gram_tokens = self.tokenizer(ent_str.replace('_', ' '))
            gram_indices = [self.gram_dict.get(token, 0) for token in gram_tokens]
            gram_indices = equalize_len(gram_indices, self.args.max_gram_size)
            ent_gram_tokens[ent_id] = gram_indices

            word_tokens = ent_str.replace('_', ' ').lower().split()
            word_indices = [self.word_dict.get(token, 0) for token in word_tokens]
            word_indices = equalize_len(word_indices, self.args.max_word_size)
            ent_word_tokens[ent_id] = word_indices

        return ent_gram_tokens, ent_word_tokens

    def _gen_wiki_mention_tokens(self):
        all_mention_gram_tokens = []
        all_mention_word_tokens = []
        all_context_word_tokens = []
        all_gold = []
        for tokens, mentions in self.data:
            for ent_idx, (mention, ent_str) in enumerate(mentions):
                if ent_str in self.ent_dict:
                    ent_id = self.ent_dict[ent_str]
                else:
                    continue

                # Gold
                all_gold.append(ent_id)

                # Mention Gram Tokens
                mention_gram_tokens = np.zeros(self.args.max_gram_size)
                mention_gram_tokens_array = np.array([self.gram_dict.get(token, 0)
                                                      for token in self.tokenizer(mention)][:self.args.max_gram_size],
                                                     dtype=np.int64)
                mention_gram_tokens[:len(mention_gram_tokens_array)] = mention_gram_tokens_array
                all_mention_gram_tokens.append(mention_gram_tokens)

                # Mention Word Tokens
                mention_word_tokens = np.zeros(self.args.max_word_size)
                mention_word_tokens_array = np.array([self.word_dict.get(token, 0)
                                                      for token in mention.lower().split()][:self.args.max_word_size],
                                                     dtype=np.int64)
                mention_word_tokens[:len(mention_word_tokens_array)] = mention_word_tokens_array
                all_mention_word_tokens.append(mention_word_tokens)

                # Context Word Tokens
                context_word_tokens = [self.word_dict[token] for token in tokens if token in self.word_dict]
                context_word_tokens = np.array(equalize_len(context_word_tokens,
                                                            self.args.max_context_size)).astype(np.int64)
                all_context_word_tokens.append(context_word_tokens)

        return all_gold, all_mention_gram_tokens, all_mention_word_tokens, all_context_word_tokens

    def _gen_conll_mention_tokens(self):
        all_context_word_tokens = []
        all_mention_word_tokens = []
        all_mention_gram_tokens = []
        all_gold = []

        for func in [is_training_doc, is_dev_doc, is_test_doc]:
            for text, gold_ents, _, _, _ in iter_docs(join(self.args.data_path, 'Conll', 'AIDA-YAGO2-dataset.tsv'), func):
                context_word_tokens = [self.rev_word_dict.get(token, 0) for token in text.lower().split()]
                context_word_tokens = equalize_len(context_word_tokens, self.args.max_context_size)
                for ent_str, (begin, end) in gold_ents:
                    if ent_str in self.ent_dict:
                        mention = text[begin:end]
                        all_gold.append(self.ent_dict[ent_str])
                        all_context_word_tokens.append(context_word_tokens)

                        mention_gram_tokens = [self.gram_dict.get(token, 0) for token in self.tokenizer(mention)]
                        mention_gram_tokens = equalize_len(mention_gram_tokens, self.args.max_gram_size)
                        all_mention_gram_tokens.append(mention_gram_tokens)

                        mention_word_tokens = [self.word_dict.get(token, 0) for token in mention.lower().split()]
                        mention_word_tokens = equalize_len(mention_word_tokens, self.args.max_word_size)
                        all_mention_word_tokens.append(mention_word_tokens)

        return all_gold, all_mention_gram_tokens, all_mention_word_tokens, all_context_word_tokens

    def validate(self,
                 model=None,
                 error=False,
                 error_size=10,
                 gram=True,
                 word=True,
                 context=True,
                 norm_gram=True,
                 norm_word=True,
                 norm_context=True,
                 norm_final=False,
                 verbose=True,
                 measure='ip'):
        model.eval()

        # Get Embeddings
        word_embs = model.state_dict()['word_embs.weight'].cpu().numpy()
        ent_embs = model.state_dict()['ent_embs.weight'].cpu().numpy()
        gram_embs = model.state_dict()['gram_embs.weight'].cpu().numpy()
        orig_W = model.state_dict()['orig_linear.weight'].cpu().numpy()
        orig_b = model.state_dict()['orig_linear.bias'].cpu().numpy()

        if gram:
            ent_gram_embs = gram_embs[self.ent_gram_indices, :]
            ent_gram_embs = ent_gram_embs.mean(axis=1)

            mention_gram_indices = self.mention_gram_indices[self.mask, :]
            mention_gram_embs = gram_embs[mention_gram_indices, :].mean(axis=1)

            if norm_gram:
                mention_gram_embs = normalize(mention_gram_embs)
                ent_gram_embs = normalize(ent_gram_embs)

        if word:
            ent_word_embs = ent_embs[self.ent_word_indices, :].mean(axis=1)
            ent_word_embs = ent_word_embs @ orig_W + orig_b

            mention_word_indices = self.mention_word_indices[self.mask, :]
            mention_word_embs = word_embs[mention_word_indices, :].mean(axis=1)
            mention_word_embs = mention_word_embs @ orig_W + orig_b

            if norm_word:
                mention_word_embs = normalize(mention_word_embs)
                ent_word_embs = normalize(ent_word_embs)

        if context:
            context_word_indices = self.context_word_indices[self.mask, :]
            context_word_embs = word_embs[context_word_indices, :].mean(axis=1)
            context_word_embs = context_word_embs @ orig_W + orig_b

            if norm_context:
                context_word_embs = normalize(context_word_embs)

        # 100
        if gram and not context and not word:
            ent_combined_embs = ent_gram_embs
            mention_combined_embs = mention_gram_embs
        # 001
        elif not gram and not context and word:
            ent_combined_embs = ent_word_embs
            mention_combined_embs = mention_word_embs
        # 010
        elif not gram and context and not word:
            ent_combined_embs = ent_embs
            mention_combined_embs = context_word_embs
        # 110
        elif gram and context and not word:
            ent_combined_embs = np.concatenate((ent_embs, ent_gram_embs), axis=1)
            mention_combined_embs = np.concatenate((context_word_embs, mention_gram_embs), axis=1)
        # 101
        elif gram and not context and word:
            ent_combined_embs = np.concatenate((ent_word_embs, ent_gram_embs), axis=1)
            mention_combined_embs = np.concatenate((mention_word_embs, mention_gram_embs), axis=1)
        # 011
        elif not gram and context and word:
            ent_combined_embs = np.concatenate((ent_embs, ent_word_embs), axis=1)
            mention_combined_embs = np.concatenate((context_word_embs, mention_word_embs), axis=1)
        # 111
        else:
            ent_combined_embs = np.concatenate((ent_embs, ent_gram_embs, ent_word_embs), axis=1)
            mention_combined_embs = np.concatenate((context_word_embs, mention_gram_embs, mention_word_embs), axis=1)

        if norm_final:
            mention_combined_embs = normalize(mention_combined_embs)
            ent_combined_embs = normalize(ent_combined_embs)

        if verbose:
            logger.info('Ent Shape : {}'.format(ent_combined_embs.shape))
            logger.info('Mention Shape : {}'.format(ent_combined_embs.shape))
            print(ent_combined_embs[:5, :])
            print(mention_combined_embs[:5, :])

        # Create / search in Faiss Index
        if verbose:
            logger.info("Searching in index")

        if measure == 'ip':
            index = faiss.IndexFlatIP(ent_combined_embs.shape[1])
        else:
            index = faiss.IndexFlatL2(mention_combined_embs.shape[1])

        D, I = index.search(mention_combined_embs.astype(np.float32), 100)
        if verbose:
            print(I[:20, :10])
            logger.info("Search Complete")

        # Error Analysis
        if error:
            for i in range(error_size):
                s = ''
                if gram:
                    m_g = mention_gram_indices[i]
                    s += ''.join([self.rev_gram_dict[token][0] for token in m_g if token in self.rev_gram_dict]) + '|'
                if word:
                    m_w = mention_word_indices[i]
                    s += ' '.join([self.rev_gram_dict[token] for token in m_w if token in self.rev_gram_dict]) + '|'
                if context:
                    c_w = context_word_indices[i][:20]
                    s += ' '.join([self.rev_word_dict[token] for token in c_w if token in self.rev_word_dict]) + '|'

                s += self.rev_ent_dict[self.all_gold[self.mask[i]]] + '>>>>>'
                s += ','.join([self.rev_ent_dict[ent_id] for ent_id in I[i][:10] if ent_id in self.rev_ent_dict])
                print(s + '\n')

        # Evaluate rankings
        if verbose:
            logger.info("Starting Evaluation of rankings")

        top1, top10, top100, mrr = eval_ranking(I, self.all_gold[self.mask.astype(np.int32)], [1, 10, 100], also_topk=True)

        return top1, top10, top100, mrr
