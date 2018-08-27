# This module implements a trainer to be used by train.py
import logging

import sys
from os.path import join

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_utils import save_checkpoint

logger = logging.getLogger()


class Trainer(object):

    def __init__(self, loader=None, args=None, model=None, validator=None, model_dir=None):
        self.loader = loader
        self.args = args
        self.model = model
        self.num_epochs = self.args.num_epochs
        self.validator = validator
        self.model_dir = model_dir
        self.min_delta = 1e-03
        self.patience = self.args.patience

        if args.optim == 'adagrad':
            self.optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=args.wd)
        elif args.optim == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=args.wd)
        elif args.optim == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                          weight_decay=args.wd)
        else:
            logger.error("Optimizer {} not recognized, choose between adam, adagrad, rmsprop".format(args.optim))
            sys.exit(1)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', verbose=True)
        self.loader_index = 0

    def _combined_get_next_batch(self, data):
        data = list(data)
        ymask = data[0]
        data = data[1:]
        for i in range(len(data)):
            data[i] = Variable(data[i])

        ymask = ymask.view(self.args.batch_size * self.args.max_ent_size)
        ymask = Variable(ymask)
        labels = Variable(torch.zeros(self.args.batch_size * self.args.max_ent_size).type(torch.LongTensor), requires_grad=False)

        if self.args.use_cuda:
            if isinstance(self.args.device, int):
                for i in range(len(data)):
                    data[i] = data[i].cuda(self.args.device)
                ymask = ymask.cuda(self.args.device)
                labels = labels.cuda(self.args.device)
            else:
                ymask = ymask.cuda(self.args.device[0])
                labels = labels.cuda(self.args.device[0])

        return tuple(data), ymask, labels

    def _yamada_get_next_batch(self, data):
        data = list(data)

        ymask = data[0]
        b, e = ymask.shape
        ymask = ymask.view(b * e)
        labels = data[1].view(b * e)
        data = data[2:]
        for i in range(len(data)):
            data[i] = Variable(data[i])
        ymask = Variable(ymask)
        labels = Variable(labels, requires_grad=False)

        if self.args.use_cuda:
            if isinstance(self.args.device, int):
                for i in range(len(data)):
                    data[i] = data[i].cuda(self.args.device)
                ymask = ymask.cuda(self.args.device)
                labels = labels.cuda(self.args.device)
            else:
                ymask = ymask.cuda(self.args.device[0])
                labels = labels.cuda(self.args.device[0])

        return tuple(data), ymask, labels

    def _get_next_batch(self, data):
        if self.args.model == 'combined':
            return self._combined_get_next_batch(data)
        elif self.args.model == 'yamada':
            return self._yamada_get_next_batch(data)
        else:
            logger.error("Model {} not recognized, choose between combined, yamada".format(self.args.model))
            sys.exit(1)

    def _cosine_loss(self, scores, ymask):
        zeros_2d = Variable(torch.zeros(self.args.batch_size * self.args.max_ent_size, self.args.num_candidates - 1))
        if self.args.use_cuda:
            if isinstance(self.args.device, int):
                zeros_2d = zeros_2d.cuda(self.args.device)
            else:
                zeros_2d = zeros_2d.cuda(self.args.device[0])
        scores_pos = scores[:, 0]
        scores_neg = scores[:, 1:]

        distance_pos = 1 - scores_pos
        distance_neg = torch.max(zeros_2d, scores_neg - self.args.margin)

        ymask_2d = ymask.repeat(self.args.num_candidates - 1).view(self.args.num_candidates - 1, -1).transpose(0, 1)
        distance_pos_masked = distance_pos * ymask
        distance_neg_masked = distance_neg * ymask_2d

        loss_pos = distance_pos_masked.sum() / ymask.sum()
        loss_neg = distance_neg_masked.sum() / ymask_2d.sum()
        loss = loss_pos + loss_neg

        return loss

    @staticmethod
    def _cross_entropy(scores, ymask, labels):
        loss = F.cross_entropy(scores, labels) * ymask
        loss = loss.sum() / ymask.sum()

        return loss

    def step(self, data):
        data, ymask, labels = self._get_next_batch(data)
        try:
            scores = self.model(data)
        except RuntimeError:
            return 0

        if self.args.loss_func == 'cosine':
            loss = self._cosine_loss(scores, ymask)
        elif self.args.loss_func == 'cross_entropy':
            loss = self._cross_entropy(scores, ymask, labels)
        else:
            logger.error("Loss function {} not recognized, choose one of cosine, cross_entropy")
            sys.exit(1)

        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def combined_validate(self, epoch):
        top1_wiki, top10_wiki, top100_wiki, mrr_wiki, top1_conll, top10_conll, top100_conll, mrr_conll = self.validator.validate(
            model=self.model)
        logger.info('Dev Validation')
        logger.info(
            "Wikipedia: Epoch {} Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(epoch,
                                                                                                         top1_wiki,
                                                                                                         top10_wiki,
                                                                                                         top100_wiki,
                                                                                                         mrr_wiki))
        logger.info(
            "Conll: Epoch {} Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(epoch,
                                                                                                     top1_conll,
                                                                                                     top10_conll,
                                                                                                     top100_conll,
                                                                                                     mrr_conll))
        return mrr_conll

    def yamada_validate(self, epoch):
        correct, mentions = self.validator.validate(model=self.model)
        perc = correct / mentions * 100
        logger.info('Epoch : {}, Correct : {}, Mention : {}, Percentage : {}'.format(epoch, correct, mentions, perc))

        return perc

    def train(self):
        training_losses = []
        best_model = self.model
        wait = 0
        best_valid_metric = 0

        for epoch in range(self.num_epochs):
            self.model = self.model.train()
            for batch_no, data in enumerate(self.loader, 0):
                if batch_no % 1000 == 0:
                    logger.info("Now on batch : {}".format(batch_no))
                loss = self.step(data)
                training_losses.append(loss)

            logger.info('Epoch - {}, Training Loss - {:.4}'.format(epoch, loss.data[0]))
            if epoch % self.args.save_every == 0 and epoch != 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, filename=join(self.model_dir, '{}.ckpt'.format(epoch)))

            if self.args.model == 'combined':
                valid_metric = self.combined_validate(epoch)
            elif self.args.model == 'yamada':
                valid_metric = self.yamada_validate(epoch)
            else:
                logger.error("Model {} not recognized, choose between combined, yamada".format(self.args.model))
                sys.exit(1)

            self.scheduler.step(valid_metric)

            if valid_metric > best_valid_metric:
                best_model = self.model
                best_valid_metric = valid_metric
                wait = 0
            else:
                if wait >= self.patience + 5:
                    logger.info("Network not improving, breaking at epoch {}".format(epoch))
                    break
                wait += 1

        save_checkpoint({
            'state_dict': best_model.state_dict(),
            'optimizer': self.optimizer.state_dict()}, filename=join(self.model_dir, 'best_model.ckpt'))

        return best_model
