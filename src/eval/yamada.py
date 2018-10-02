# Validator class for yamada model
import numpy as np
from torch.autograd import Variable
from torch.nn import DataParallel
from logging import getLogger

from src.utils.utils import reverse_dict


logger = getLogger()


class YamadaValidator:
    def __init__(self, loader=None, args=None, ent_dict=None, word_dict=None):

        self.loader = loader
        self.args = args
        self.ent_dict = ent_dict
        self.word_dict = word_dict
        self.rev_ent_dict = reverse_dict(ent_dict)
        self.rev_word_dict = reverse_dict(word_dict)

    def _get_next_batch(self, data):
        data = list(data)
        ymask = data[0]
        b, e = ymask.shape
        data = data[1:]
        for i in range(len(data)):
            data[i] = Variable(data[i])

        ymask = ymask.view(b * e)
        labels = np.zeros_like(ymask)

        if self.args.use_cuda:
            if isinstance(self.args.device, int):
                for i in range(len(data)):
                    data[i] = data[i].cuda(self.args.device)

        return tuple(data), ymask, labels

    def validate(self, model):
        model = model.eval()
        model.cpu()

        total_correct = 0
        total_mention = 0
        errors = []

        for batch_no, data in enumerate(self.loader, 0):
            data, ymask, labels = self._get_next_batch(data)

            context, candidates = data[:2]
            context, candidates = context.data.numpy(), candidates.data.numpy()

            scores = model(data)
            scores = scores.data.numpy()

            preds = np.argmax(scores, axis=1)
            print(f'PREDS : {preds}')
            correct = (np.equal(preds, labels) * ymask).sum()
            print(f'CORRECT : {correct}')
            inc = np.not_equal(preds, labels) * ymask
            inc_ids = np.where(inc)
            print(f'INCORRECT IDS : {inc_ids}')

            context_str = ''
            pred_str = ''
            for inc_i in inc_ids:
                word_tokens = context[inc_i]
                context_str += ' '.join([self.rev_word_dict.get(word_token, '') for word_token in word_tokens[:10]])
                pred_ids = -preds[inc_i].argsort()
                pred_str += ','.join([self.rev_ent_dict.get(pred_id, '') for pred_id in pred_ids])
            print(f'CONTEXT : {context_str}')
            print(f'Pred : {pred_str}')

            mention = ymask.sum()
            total_correct += correct
            total_mention += mention

        if self.args.use_cuda:
            if isinstance(self.args.device, tuple):
                model.cuda(self.args.device[0])
                DataParallel(model, self.args.device)
            else:
                model.cuda(self.args.device)

        return total_correct, total_mention

