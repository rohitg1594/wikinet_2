# Validator class for yamada model
import numpy as np
from torch.autograd import Variable
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
        for i in range(len(data)):
            data[i] = Variable(data[i])

        labels = np.zeros(self.args.batch_size)

        if self.args.use_cuda:
            if isinstance(self.args.device, int):
                for i in range(len(data)):
                    data[i] = data[i].cuda(self.args.device)

        return tuple(data), labels

    def validate(self, model):
        model = model.eval()

        total_correct = 0
        total_mention = 0

        for batch_no, data in enumerate(self.loader, 0):
            data, labels = self._get_next_batch(data)

            context, candidates = data[:2]
            context, candidates = context.cpu().data.numpy(), candidates.cpu().data.numpy()

            scores = model(data)
            scores = scores.cpu().data.numpy()

            preds = np.argmax(scores, axis=1)
            print(f'PREDS : {preds}')
            correct = (np.equal(preds, labels)).sum()
            print(f'CORRECT : {correct}')
            inc = np.not_equal(preds, labels)
            inc_ids = np.where(inc)
            print(f'INCORRECT IDS : {inc_ids}')

            #context_str = ''
            #pred_str = ''
            #for inc_i in inc_ids:
            #    word_tokens = context[inc_i]
            #    context_str += ' '.join([self.rev_word_dict.get(word_token, '') for word_token in word_tokens[:10]])
            #    pred_ids = -preds[inc_i].argsort()
            #    pred_str += ','.join([self.rev_ent_dict.get(pred_id, '') for pred_id in pred_ids])
            #print(f'CONTEXT : {context_str}')
            #print(f'Pred : {pred_str}')

            total_correct += correct
            total_mention += scores.shape[0]

        return total_correct, total_mention

