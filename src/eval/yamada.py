# Validator class for yamada model
from os.path import join

import numpy as np
from torch.autograd import Variable
from logging import getLogger

from src.utils.utils import reverse_dict


logger = getLogger()


class YamadaValidator:
    def __init__(self, loader=None, args=None, ent_dict=None, word_dict=None, data_type=None):

        self.loader = loader
        self.args = args
        self.ent_dict = ent_dict
        self.word_dict = word_dict
        self.rev_ent_dict = reverse_dict(ent_dict)
        self.rev_word_dict = reverse_dict(word_dict)
        self.data_type = data_type

    def _get_next_batch(self, data):
        data = list(data)
        for i in range(len(data)):
            data[i] = Variable(data[i])

        labels = np.zeros(data[0].shape[0])

        if self.args.use_cuda:
            if isinstance(self.args.device, int):
                for i in range(len(data)):
                    data[i] = data[i].cuda(self.args.device)

        return tuple(data), labels

    def get_pred_str(self, ids, context, preds, candidates):

        comp_str = ''
        for id in ids:
            word_tokens = context[id]
            context_str = ' '.join([self.rev_word_dict.get(word_token, '') for word_token in word_tokens[:20]])
            pred_ids = (-preds[id]).argsort()
            print(f'pred ids:{pred_ids}')
            pred_str = ','.join([self.rev_ent_dict.get(pred_id, '') for pred_id in pred_ids])
            correct_ent = self.rev_ent_dict.get(candidates[id][0], '')
            comp_str += '||'.join([correct_ent, pred_str, context_str]) + '\n'

        return comp_str

    def validate(self, model):
        model = model.eval()

        total_correct = 0
        total_mention = 0
        cor_pred_str = ''
        inc_pred_str = ''

        for batch_no, data in enumerate(self.loader, 0):
            data, labels = self._get_next_batch(data)
            scores, _, _ = model(data)
            scores = scores.cpu().data.numpy()

            preds = np.argmax(scores, axis=1)
            num_cor = (np.equal(preds, labels)).sum()

            cor = np.equal(preds, labels)
            inc = np.not_equal(preds, labels)
            inc_ids = np.where(inc)[0]
            cor_ids = np.where(cor)[0]

            context, candidates = data[:2]
            context, candidates = context.cpu().data.numpy(), candidates.cpu().data.numpy()
            inc_pred_str += self.get_pred_str(inc_ids, context, preds, candidates)
            cor_pred_str += self.get_pred_str(cor_ids, context, preds, candidates)

            total_correct += num_cor
            total_mention += scores.shape[0]

        with open(join(self.args.model_dir, f'inc_preds_{self.data_type}.txt'), 'w') as f:
            f.write(inc_pred_str)

        with open(join(self.args.model_dir, f'cor_preds_{self.data_type}.txt'), 'w') as f:
            f.write(cor_pred_str)

        return total_correct, total_mention

