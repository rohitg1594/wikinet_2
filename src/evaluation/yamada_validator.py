# Validator class for yamada model
import numpy as np

from torch.autograd import Variable

from logging import getLogger

logger = getLogger()


class YamadaValidator:
    def __init__(self, loader=None, args=None):

        self.loader = loader
        self.args = args

    def _get_next_batch(self, data):
        data = list(data)
        ymask = data[0].numpy()
        labels = data[1].numpy()
        data = data[2:]
        for i in range(len(data)):
            data[i] = Variable(data[i])

        if self.args.use_cuda:
            for i in range(len(data)):
                data[i] = data[i].cuda(self.args.device)
            ymask = ymask.cuda(self.args.device)
            labels = labels.cuda(self.args.device)

        return tuple(data), ymask, labels

    def validate(self, model):
        model.eval()

        total_correct = 0
        total_mention = 0

        for batch_no, data in enumerate(self.loader, 0):
            data, ymask, labels = self._get_next_batch(data)

            scores, out_vecs = model(data)
            scores = scores.cpu().data.numpy()

            ymask = ymask.reshape(self.args.batch_size * self.args.max_ent_size, -1)
            labels = labels.reshape(self.args.batch_size * self.args.max_ent_size, -1)
            preds = np.argmax(scores, axis=1)
            correct = (np.equal(preds, labels) * ymask).sum()
            mention = ymask.sum()

            total_correct += correct
            total_mention += mention

        return total_correct, total_mention
