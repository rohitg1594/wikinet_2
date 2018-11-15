# Validator class
import faiss
from logging import getLogger

from src.utils.utils import *
from src.tokenizer.regexp_tokenizer import RegexpTokenizer

logger = getLogger(__name__)

RE_DOCID = re.compile('^\d+')


class CombinedValidator:
    def __init__(self,
                 gram_tokenizer=None,
                 gram_dict=None,
                 word2id=None,
                 ent2id=None,
                 data=None,
                 args=None,):

        self.gram_tokenizer = gram_tokenizer
        self.word_tokenizer = RegexpTokenizer()
        self.word2id = word2id
        self.id2word = reverse_dict(self.word2id)
        self.ent2id = ent2id
        self.id2ent = reverse_dict(self.ent2id)
        self.gram2id = gram_dict
        self.id2gram = reverse_dict(self.gram2id)
        self.data = data
        self.args = args
        self.model_name = self.args.model_name

        autoencoder_data = pickle_load(join(self.args.data_path, 'autoencoder/data.pickle'))
        self.char_dict = autoencoder_data['char_dict']
        self.max_char_size = self.args.max_char_size

        # Get entity tokens
        logger.info('Creating ent tokens.....')
        self.ent_gram_indices, self.ent_word_indices, self.ent_char_indices = self._get_ent_tokens()
        logger.info('Created ent tokens.')

        # Get wiki, msnbc, ace2004 and conll mention tokens
        self.numpy_data = {}
        self.data_types = ['wiki', 'msnbc', 'conll', 'ace2004']
        for data_type in self.data_types:
            logger.info(f'Numpifying {data_type} dataset.....')
            self.numpy_data[data_type] = self._numpify_data(data_type=data_type)
            logger.info('Numpified.')

        # Mask to select wiki mention queries
        self.wiki_mask = np.random.choice(np.arange(len(self.numpy_data['wiki']['mention_gram'])),
                                          size=self.args.query_size).astype(np.int32)

    def _get_ent_tokens(self):
        """Creates numpy arrays containing gram and word token ids for each entity."""
        out_f = join(self.args.data_path, 'cache', 'eval_ent_tokens')
        if os.path.exists(out_f):
            logger.info(f"Loading eval ent tokens from {out_f}")
            ent_tokens_dict = pickle_load(out_f)
            ent_gram_indices = ent_tokens_dict['ent_gram_indices']
            ent_word_indices = ent_tokens_dict['ent_word_indices']
            ent_char_indices = ent_tokens_dict['ent_char_indices']
        else:
            # Init Tokens
            ent_gram_indices = np.zeros((len(self.ent2id) + 1, self.args.max_gram_size)).astype(np.int32)
            ent_word_indices = np.zeros((len(self.ent2id) + 1, self.args.max_word_size)).astype(np.int32)
            ent_char_indices = np.zeros((len(self.ent2id) + 1, self.args.max_char_size)).astype(np.int32)

            # For each entity
            for ent_str, ent_id in self.ent2id.items():

                # Remove underscore
                ent_str = ent_str.replace('_', ' ')

                # Gram tokens
                gram_tokens = self.gram_tokenizer(ent_str)
                gram_indices = [self.gram2id.get(token, 0) for token in gram_tokens]
                gram_indices = equalize_len(gram_indices, self.args.max_gram_size)
                ent_gram_indices[ent_id] = gram_indices

                # Word tokens
                word_tokens = [token.text.lower() for token in self.word_tokenizer.tokenize(ent_str)]
                word_indices = [self.word2id.get(token, 0) for token in word_tokens]
                word_indices = equalize_len(word_indices, self.args.max_word_size)
                ent_word_indices[ent_id] = word_indices

                # Char tokens
                char_tokens = list(ent_str)
                char_indices = [self.char_dict[token] for token in char_tokens]
                char_indices = equalize_len_w_eot(char_indices, self.args.max_char_size, self.char_dict['EOT'])
                ent_char_indices[ent_id] = char_indices

            with open(out_f, 'wb') as f:
                ent_tokens_dict = {'ent_gram_indices': ent_gram_indices,
                                   'ent_word_indices': ent_word_indices,
                                   'ent_char_indices': ent_char_indices}
                pickle.dump(ent_tokens_dict, f)

        return ent_gram_indices, ent_word_indices, ent_char_indices

    def _numpify_data(self, data_type='wiki'):
        """ Creates numpy arrays containing gram and word token ids or each mention and
        word tokens for context in abstract. Also output gold entity labels. Out put is a dictionary."""

        out_f = join(self.args.data_path, 'cache', f'eval_mention_tokens_{data_type}')
        if os.path.exists(out_f):
            logger.info(f"Loading eval mention tokens from {out_f}")
            output = pickle_load(out_f)
        else:

            # Init lists
            all_mention_gram_indices = []
            all_mention_word_indices = []
            all_mention_char_indices = []
            all_context_word_indices = []
            all_small_context_indices = []
            all_gold = []

            if data_type == 'wiki':
                data = self.data
            elif data_type == 'msnbc':
                data = pickle_load(join(self.args.data_path, 'training_files', 'msnbc.pickle'))
            elif data_type == 'ace2004':
                data = pickle_load(join(self.args.data_path, 'training_files', 'ace2004.pickle'))
            elif data_type == 'conll':
                data = pickle_load(join(self.args.data_path, 'training_files', f'conll-{self.args.conll_split}.pickle'))
            else:
                logger.info('Data type not understood, choose one of msnbc, wiki, ace2004 and conll')
                sys.exit(1)

            id2context, examples = data

            # For each abstract
            for example in examples:
                context_id, (mention, ent_str, mention_char_span, small_context_tokens) = example

                # Check if entity is relevant
                ent_id = self.ent2id.get(ent_str, 0)

                # Gold
                all_gold.append(ent_id)

                # Mention Gram
                mention_gram_tokens = [token for token in self.gram_tokenizer(mention)]
                mention_gram_indices = [self.gram2id.get(token, 0) for token in mention_gram_tokens]
                mention_gram_indices = equalize_len(mention_gram_indices, self.args.max_gram_size)
                all_mention_gram_indices.append(np.array(mention_gram_indices).astype(np.int64))

                # Mention Word
                mention_word_tokens = [token.text.lower() for token in self.word_tokenizer.tokenize(mention)]
                mention_word_indices = [self.word2id.get(token, 0) for token in mention_word_tokens]
                mention_word_indices = equalize_len(mention_word_indices, self.args.max_word_size)
                all_mention_word_indices.append(np.array(mention_word_indices).astype(np.int64))

                # Mention Char
                mention_char_tokens = list(mention)
                mention_char_indices = [self.char_dict[token] for token in mention_char_tokens]
                mention_char_indices = equalize_len_w_eot(mention_char_indices, self.args.max_char_size, self.char_dict['EOT'])
                all_mention_char_indices.append(np.array(mention_char_indices).astype(np.int64))

                # Context Word
                context_word_indices = [self.word2id.get(token, 0) for token in id2context[context_id]]
                context_word_indices = equalize_len(context_word_indices, self.args.max_context_size)
                all_context_word_indices.append(np.array(context_word_indices).astype(np.int64))

                # Small Context
                all_small_context_indices.append(small_context_tokens)

            output = {'gold': np.array(all_gold).astype(np.int32),
                      'mention_gram': np.vstack(all_mention_gram_indices).astype(np.int32),
                      'mention_word': np.vstack(all_mention_word_indices).astype(np.int32),
                      'mention_char': np.vstack(all_mention_char_indices).astype(np.int32),
                      'context': np.vstack(all_context_word_indices).astype(np.int32),
                      'small_context': np.vstack(all_small_context_indices).astype(np.int32)
                      }

            with open(out_f, 'wb') as f:
                pickle.dump(output, f)

        return output

    def _get_data(self, data_type='wiki', cuda=False):

        ent_gram = torch.from_numpy(self.ent_gram_indices).long()
        ent_word = torch.from_numpy(self.ent_word_indices).long()
        ent_char = torch.from_numpy(self.ent_char_indices).long()
        ent_ids = torch.arange(0, len(self.ent2id) + 1).long()

        mention_gram = torch.from_numpy(self.numpy_data[data_type]['mention_gram']).long()
        mention_word = torch.from_numpy(self.numpy_data[data_type]['mention_word']).long()
        mention_char = torch.from_numpy(self.numpy_data[data_type]['mention_char']).long()

        context = torch.from_numpy(self.numpy_data[data_type]['context']).long()
        small_context = torch.from_numpy(self.numpy_data[data_type]['small_context']).long()

        if data_type == 'wiki':
            mention_gram = mention_gram[self.wiki_mask, :]
            mention_word = mention_word[self.wiki_mask, :]
            mention_char = mention_char[self.wiki_mask, :]
            context = context[self.wiki_mask, :]
            small_context = small_context[self.wiki_mask, :]

        total_data = {'mention_word_tokens': mention_word,
                      'mention_char_tokens': mention_char,
                      'mention_gram_tokens': mention_gram,
                      'candidate_word_tokens': ent_word,
                      'candidate_char_tokens': ent_char,
                      'candidate_gram_tokens': ent_gram,
                      'candidate_ids': ent_ids,
                      'context_tokens': context,
                      'small_context': small_context,
                      }

        if self.model_name == 'weigh_concat':
            keys = ['mention_gram_tokens', 'context_tokens', 'candidate_gram_tokens', 'candidate_ids']
        elif self.model_name == 'mention_prior':
            keys = ['mention_gram_tokens', 'mention_word_tokens', 'context_tokens', 'candidate_gram_tokens',
                    'candidate_ids']
        elif self.model_name in ['average', 'linear', 'multi_linear', 'rnn']:
            keys = ['mention_word_tokens', 'candidate_ids']
        elif self.model_name == 'with_string':
            keys = ['mention_gram_tokens', 'mention_word_tokens', 'candidate_gram_tokens', 'candidate_ids']
        elif self.model_name == 'small_context':
            keys = ['mention_word_tokens', 'small_context', 'candidate_ids']
        elif self.model_name in ['full_context', 'full_context']:
            keys = ['mention_word_tokens', 'context_tokens', 'candidate_ids']
        elif self.model_name == 'full_context_string':
            keys = ['mention_word_tokens', 'mention_char_tokens', 'context_tokens', 'candidate_char_tokens',
                    'candidate_ids']
        elif self.model_name == 'pre_train':
            keys = ['context_tokens', 'candidate_ids']
        else:
            logger.error(f'model {self.args.model_name} not implemented')
            sys.exit(1)

        data = {k: total_data[k] for k in keys}

        return data

    def _get_debug_string(self, preds=None, data_type='wiki', result=False):

        mention_gram = self.numpy_data[data_type]['mention_gram']
        mention_word = self.numpy_data[data_type]['mention_word']
        context = self.numpy_data[data_type]['context']
        small_context = self.numpy_data[data_type]['small_context']
        gold = self.numpy_data[data_type]['gold']

        s = ''
        for i in range(10):
            # m_g = mention_gram[i]
            # s += ''.join([self.rev_gram_dict[token][0] for token in m_g if token in self.rev_gram_dict]) + '|'
            m_w = mention_word[i]
            s += ' '.join([self.id2word[token] for token in m_w if token in self.id2word]) + '|'
            c_w = small_context[i][:20]
            s += ' '.join([self.id2word[token] for token in c_w if token in self.id2word]) + '|'
            # c_w = context[i][:20]
            # s += ' '.join([self.rev_word_dict[token] for token in c_w if token in self.rev_word_dict]) + '|'
            s += self.id2ent[gold[i]] + '|'
            if result:
                s += ','.join([self.id2ent[ent_id] for ent_id in preds[i][:10] if ent_id in self.id2ent])
            s += '\n'

        return s

    def validate(self, model=None, error=True):
        model.eval()
        model = model.cpu()
        assert not model.training
        first_data = True
        results = {}

        for data_type in self.data_types:
            input = self._get_data(data_type=data_type, cuda=True)
            scores, ent_combined_embs, mention_combined_embs = model(input)

            mention_combined_embs = mention_combined_embs.data.numpy()
            ent_combined_embs = ent_combined_embs.data.numpy()

            if first_data:
                # Create / search in Faiss Index
                if self.args.measure == 'ip':
                    index = faiss.IndexFlatIP(ent_combined_embs.shape[1])
                    logger.info("Using IndexFlatIP")
                else:
                    index = faiss.IndexFlatL2(ent_combined_embs.shape[1])
                    logger.info("Using IndexFlatL2")
                index.add(ent_combined_embs)
                first_data = False

            logger.info(f"Searching in index with query size : {mention_combined_embs.shape}.....")
            _, preds = index.search(mention_combined_embs.astype(np.float32), 100)
            logger.info("Search complete.")

            # Evaluate rankings
            gold = self.numpy_data[data_type]['gold']
            gold = gold[self.wiki_mask] if data_type == 'wiki' else gold
            top1, top10, top100, mrr = eval_ranking(preds, gold, [1, 10, 100])
            results[data_type] = {'top1': top1,
                                  'top10': top10,
                                  'top100': top100,
                                  'mrr': mrr}

            # Print combination weights
            if self.args.model_name == "full_context_string":
                for k, v in model.state_dict().items():
                    if isinstance(v, torch.nn.parameter.Parameter):
                        print(k, v)

            # Error analysis
            if error:
                print(f'{data_type.upper()}\n')
                mention_gram = self.numpy_data[data_type]['mention_gram']
                mention_gram = mention_gram[self.wiki_mask, :] if data_type == 'wiki' else mention_gram
                check_errors(preds, gold, mention_gram, self.id2ent, self.id2gram, [1, 10, 100])
                print()
        if self.args.use_cuda:
            model = send_to_cuda(self.args.device, model)

        return results
