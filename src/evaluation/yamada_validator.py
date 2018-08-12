# Validator class for yamada model
import numpy as np

from torch.autograd import Variable

from logging import getLogger
from collections import OrderedDict

from src.utils import normalize

logger = getLogger()

np.set_printoptions(threshold=10 ** 6)

class YamadaValidator:
    def __init__(self, loader=None, args=None):

        self.loader = loader
        self.args = args

    def get_next_batch(self, data):
        data = list(data)

        ymask = data[0].numpy()
        b, e = ymask.shape
        ymask = ymask.reshape(b * e)
        labels = data[1].numpy().reshape(b * e)
        data = data[2:]
        for i in range(len(data)):
            data[i] = Variable(data[i])

        if self.args.use_cuda:
            for i in range(len(data)):
                data[i] = data[i].cuda(self.args.device)

        return tuple(data), ymask, labels

    def validate(self, model):
        model = model.eval()

        total_correct = 0
        total_mention = 0

        for batch_no, data in enumerate(self.loader, 0):
            data, ymask, labels = self.get_next_batch(data)

            scores = model(data)
            scores = scores.cpu().data.numpy()

            preds = np.argmax(scores, axis=1)
            correct = (np.equal(preds, labels) * ymask).sum()
            mention = ymask.sum()
            total_correct += correct
            total_mention += mention

        return total_correct, total_mention

    def full_validated(self, model, dev_data):
        params = dict()

        new_state_dict = OrderedDict()
        for k, v in model.state_dict().items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v

        word_embs = new_state_dict['embeddings_word.weight'].cpu().numpy()
        ent_embs = new_state_dict['embeddings_ent.weight'].cpu().numpy()
        orig_W= new_state_dict['orig_linear.weight'].cpu().numpy()
        orig_b = new_state_dict['orig_linear.bias'].cpu().numpy()
        hidden_W = new_state_dict['hidden.weight'].cpu().numpy()
        hidden_b = new_state_dict['hidden.bias'].cpu().numpy()
        output_W = new_state_dict['output.weight'].cpu().numpy()
        output_b = new_state_dict['output.bias'].cpu().numpy()

        context_list = []
        for word_ids, examples in dev_data:
            context_vec = normalize(word_embs[word_ids].mean(axis=0) @ orig_W + orig_b)
            for _ in range(len(examples)):
                context_list.append(context_vec)
        context_matr = np.vstack(context_list)
        logger.info("Shape of context matrix : {}".info(context_matr.shape))
        dot_products = context_matr @ ent_embs
        #query_vec =


