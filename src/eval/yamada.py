# Validator class for yamada model
from os.path import join

import numpy as np
from torch.autograd import Variable
from logging import getLogger

from src.utils.utils import reverse_dict

np.set_printoptions(threshold=10**6)

logger = getLogger()


class YamadaValidator:
    def __init__(self, loader=None, args=None, ent_dict=None, word_dict=None, data_type=None, run=None):

        self.loader = loader
        self.args = args
        self.ent_dict = ent_dict
        self.word_dict = word_dict
        self.rev_ent_dict = reverse_dict(ent_dict)
        self.rev_word_dict = reverse_dict(word_dict)
        self.data_type = data_type
        self.run = run

    def _get_next_batch(self, data_dict):
        skip_keys = ['ent_strs', 'cand_strs', 'not_in_cand']
        for k, v in data_dict.items():
            try:
                if k not in skip_keys:
                    data_dict[k] = Variable(v)
            except:
                print(f'key - {k}, Value - {v}')

        ent_strs, cand_strs, not_in_cand = np.array(data_dict['ent_strs']),\
                                           np.array(data_dict['cand_strs']).T,\
                                           np.array(data_dict['not_in_cand'])
        for k in skip_keys:
            data_dict.pop(k)

        if self.args.use_cuda:
            device = self.args.device if isinstance(self.args.device, int) else self.args.device[0]
            for k, v in data_dict.items():
                data_dict[k] = v.cuda(device)

        return data_dict, ent_strs, cand_strs, not_in_cand

    def get_pred_str(self, batch_no, ids, context, scores, cand_strs, ent_strs):

        comp_str = ''
        for id in ids:
            word_tokens = context[id]
            mention_id = str(batch_no * self.args.batch_size + id)
            context_str = ' '.join([self.rev_word_dict.get(word_token, 'UNK_WORD') for word_token in word_tokens[:50]])
            pred_str = ','.join(cand_strs[id][(-scores[id]).argsort()][:10])
            comp_str += '||'.join([mention_id, ent_strs[id], pred_str, context_str]) + '\n'

        return comp_str

    def validate(self, model):
        model = model.eval()

        total_correct = 0
        total_not_in_cand = 0
        total_mentions = 0
        cor_adjust = 0
        cor_pred_str = ''
        inc_pred_str = ''

        for batch_no, data in enumerate(self.loader, 0):
            data_dict, ent_strs, cand_strs, not_in_cand = self._get_next_batch(data)
            # print(f'ENT STRS: {ent_strs.shape}, {ent_strs[:5]}, CAND STRS: {cand_strs.shape}, {cand_strs[:5]}')
            scores, _, _ = model(data_dict)
            scores = scores.cpu().data.numpy()

            context = data_dict['context']
            cand_ids = data_dict['cand_ids']
            context, candidates = context.cpu().data.numpy(), cand_ids.cpu().data.numpy()

            print(f'SCORES : \n {scores}')
            preds_mask = np.argmax(scores, axis=1)
            preds = cand_strs[np.arange(len(preds_mask)), preds_mask]
            # print(f'PREDS: {preds}, ENTS: {ent_strs}')

            cor = preds == ent_strs
            inc = preds != ent_strs
            num_cor = cor.sum()
            # print(f'COR: {cor}, INC: {inc}, NUM COR: {num_cor}')
            inc_idxs = np.where(inc)[0]
            cor_idxs = np.where(cor)[0]

            inc_pred_str += self.get_pred_str(batch_no, inc_idxs, context, scores, cand_strs, ent_strs)
            cor_pred_str += self.get_pred_str(batch_no, cor_idxs, context, scores, cand_strs, ent_strs)

            total_correct += num_cor
            total_mentions += scores.shape[0]
            total_not_in_cand += not_in_cand.sum()
            cor_adjust += not_in_cand[cor_idxs].sum()

        # if self.data_type == 'conll':
        #     print(f'INCORRECT PRED STR : \n {inc_pred_str[:2000]}')
        #     print('\n\n\n\n#################################################\n\n\n')
        #     print(f'CORRECT PRED STR : \n {cor_pred_str[:2000]}')

        with open(join(self.args.model_dir, f'inc_preds_{self.data_type}_{self.run}.txt'), 'w') as f:
            f.write(inc_pred_str)

        with open(join(self.args.model_dir, f'cor_preds_{self.data_type}_{self.run}.txt'), 'w') as f:
            f.write(cor_pred_str)

        return total_mentions, total_not_in_cand, total_correct, cor_adjust

