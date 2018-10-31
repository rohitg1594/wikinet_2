# Validator class for yamada model
from os.path import join

import numpy as np
from torch.autograd import Variable
from logging import getLogger

from src.utils.utils import reverse_dict


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
        for k, v in data_dict.items():
            data_dict[k] = Variable(v)

        labels = np.zeros(v.shape[0])

        if self.args.use_cuda:
            device = self.arge.device if isinstance(self.args.device, int) else self.args.device[0]
            for k, v in data_dict.items():
                data_dict[k] = v.cuda(device)

        return data_dict, labels

    def get_pred_str(self, batch_no, ids, context, scores, candidates):

        comp_str = ''
        for id in ids:
            word_tokens = context[id]
            mention_id = str(batch_no * self.args.batch_size + id)
            context_str = ' '.join([self.rev_word_dict.get(word_token, 'UNK_WORD') for word_token in word_tokens[:20]])
            pred_ids = candidates[id][(-scores[id]).argsort()][:10]
            pred_str = ','.join([self.rev_ent_dict.get(pred_id, 'UNK_ENT') for pred_id in pred_ids])
            correct_ent = self.rev_ent_dict.get(candidates[id][0], 'UNK_ENT')
            comp_str += '||'.join([mention_id, correct_ent, pred_str, context_str]) + '\n'

        return comp_str

    def validate(self, model):
        model = model.eval()

        total_correct = 0
        total_in_dict = 0
        total_not_in_cand = 0
        total_ent_ignore = 0
        cor_pred_str = ''
        inc_pred_str = ''

        for batch_no, data in enumerate(self.loader, 0):
            data, labels = self._get_next_batch(data)
            scores, _, _ = model(data)
            scores = scores.cpu().data.numpy()

            ent_ignore, true_not_in_cand, context, candidates = data[:4]
            ent_ignore, true_not_in_cand, context, candidates = ent_ignore.cpu().data.numpy(), \
                                                                true_not_in_cand.cpu().data.numpy(), \
                                                                context.cpu().data.numpy(), \
                                                                candidates.cpu().data.numpy()
            total_ent_ignore += ent_ignore.sum()

            scores = scores[ent_ignore != 1]
            labels = labels[ent_ignore != 1]
            context = context[ent_ignore != 1]
            candidates = candidates[ent_ignore != 1]
            true_not_in_cand = true_not_in_cand[ent_ignore != 1]

            preds = np.argmax(scores, axis=1)
            num_cor = (np.equal(preds, labels)).sum()

            cor = np.equal(preds, labels)
            inc = np.not_equal(preds, labels)
            inc_ids = np.where(inc)[0]
            cor_ids = np.where(cor)[0]

            inc_pred_str += self.get_pred_str(batch_no, inc_ids, context, scores, candidates)
            cor_pred_str += self.get_pred_str(batch_no, cor_ids, context, scores, candidates)

            total_correct += num_cor
            total_in_dict += scores.shape[0]
            total_not_in_cand += true_not_in_cand[cor_ids].sum()

        with open(join(self.args.model_dir, f'inc_preds_{self.data_type}_{self.run}.txt'), 'w') as f:
            f.write(inc_pred_str)

        with open(join(self.args.model_dir, f'cor_preds_{self.data_type}_{self.run}.txt'), 'w') as f:
            f.write(cor_pred_str)

        return total_in_dict, total_ent_ignore, total_not_in_cand, total_correct

