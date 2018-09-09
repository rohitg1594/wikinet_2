# This module contains functions to create training examples for original Yamada model with PERSHINA candidates.
from os.path import join
import re
from collections import defaultdict

from src.utils.utils import reverse_dict
from src.conll.iter_docs import is_training_doc, is_test_doc, is_dev_doc, iter_docs
from src.tokenizer.regexp_tokenizer import RegexpTokenizer

RE_DOCID = re.compile('^\d+')
RE_WIKI_ENT = re.compile(r'.*wiki\/(.*)')


class PershinaExamples(object):

    def __init__(self, args, yamada_model):
        self.args = args
        self.ent_dict = yamada_model['ent_dict']
        self.ent_rev = reverse_dict(self.ent_dict)
        self.word_dict = yamada_model['word_dict']
        self.tokenizer = RegexpTokenizer()

    def _get_doc_tokens(self):

        docid2tokens = {}
        for func in [is_training_doc, is_dev_doc, is_test_doc]:
            for i, (text, gold_ents, num_tokens, proper_mentions, doc_id_str) in enumerate(
                    iter_docs(join(self.args.data_path, 'Conll',
                                   'AIDA-YAGO2-dataset.tsv'), func)):
                doc_id = int(RE_DOCID.match(doc_id_str).group(0))
                tokens = self.tokenizer.tokenize(text)
                docid2tokens[doc_id] = [self.word_dict.get(token.lower(), 0) for token in tokens]

        return docid2tokens

    def _get_doc_candidates(self):

        docid2candidates = defaultdict(dict)

        for i in range(1, 1394):
            with open(join(self.args.data_path, 'Conll', 'PPRforNED', 'AIDA_candidates', 'combined', str(i)), 'r') as f:
                for line in f:

                    line = line.strip()
                    parts = line.split('\t')

                    if 'ENTITY' in line:
                        mention = parts[1][5:]
                        doc_id = int(parts[6][6:])
                        docid2candidates[doc_id][mention] = {}
                        docid2candidates[doc_id][mention]['ignore'] = 0
                        docid2candidates[doc_id][mention]['cands'] = []

                        correct_url = parts[-1][4:]
                        if correct_url == 'NIL':
                            docid2candidates[doc_id][mention]['ignore'] = 1

                    else:
                        wiki_url = parts[5]
                        ent_str = RE_WIKI_ENT.match(wiki_url).group(1)
                        if ent_str not in self.ent_dict:
                            continue
                        docid2candidates[doc_id][mention]['cands'].append(ent_str)

        return docid2candidates

    def get_training_examples(self):

        training_examples = []
        dev_examples = []
        test_examples = []

        docid2tokens = self._get_doc_tokens()
        docid2candidates = self._get_doc_candidates()

        for docid, tokenids in docid2tokens.items():
            mention_cand_tup = []
            for mention, ignore_cand_dict in docid2candidates[docid].items():
                ignore = docid2candidates[docid][mention]['ignore']
                if ignore or docid2candidates[docid][mention]['cands'] == []:
                    continue
                else:
                    mention_cand_tup.append((mention, docid2candidates[docid][mention]['cands']))

            if 1 <= docid < 947:
                training_examples.append((tokenids, mention_cand_tup))
            elif 947 <= docid < 1163:
                dev_examples.append((tokenids, mention_cand_tup))
            else:
                test_examples.append((tokenids, mention_cand_tup))

        return training_examples, dev_examples, test_examples
