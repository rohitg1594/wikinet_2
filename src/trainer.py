# This module implements a trainer to be used by train.py
import logging

import sys
from os.path import join

import torch
from torch.autograd import Variable

from src.data_utils import save_checkpoint

logger = logging.getLogger()


class Trainer(object):

    def __init__(self, loader=None, args=None, model=None, validator=None, use_cuda=False, model_dir=None):
        self.loader = loader
        self.args = args
        self.model = model
        self.use_cuda = use_cuda
        self.num_epochs = self.args.num_epochs
        self.validator = validator
        self.model_dir = model_dir

        if args.optim == 'adagrad':
            self.optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=args.wd)
        elif args.optim == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=args.wd)
        else:
            logger.error("Optimizer {} not recognized, choose between adam, adagrad".format(args.optim))
            sys.exit(1)

        self.best_valid_metric = 1e-12
        self.loader_index = 0

    def _get_next_batch(self, data):
        data = list(data)
        ymask = data[0]
        data = data[1:]
        for i in range(len(data)):
            data[i] = Variable(data[i])

        ymask = ymask.view(self.args.batch_size * self.args.max_ent_size)
        ymask = Variable(ymask)

        if self.use_cuda:
            for i in range(len(data)):
                data[i] = data[i].cuda(self.args.device)
            ymask = ymask.cuda(self.args.device)

        return tuple(data), ymask

    def _cosine_loss(self, scores, ymask):
        zeros_2d = Variable(torch.zeros(self.args.batch_size * self.args.max_ent_size, self.args.num_candidates - 1))
        if self.use_cuda:
            zeros_2d = zeros_2d.cuda(self.args.device)
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

    def train(self):
        training_losses = []
        for epoch in range(self.num_epochs):
            self.model.train()
            for i, data in enumerate(self.loader, 0):
                data, ymask = self._get_next_batch(data)
                scores = self.model(data)
                loss = self._cosine_loss(scores, ymask)

                loss.backward()
                self.optimizer.step()
                training_losses.append(loss.data[0])

            logger.info('Epoch - {}, Loss - {:.4}'.format(epoch, loss.data[0]))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}, True, filename=join(self.model_dir, '{}.ckpt'.format(epoch)))
            top1_wiki, top10_wiki, top100_wiki, mrr_wiki, top1_conll, top10_conll, top100_conll, mrr_conll = self.validator.validate(model=self.model)
            logger.info(
                "Wikipedia, Epoch - {} Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(epoch,
                                                                                                               top1_wiki,
                                                                                                               top10_wiki,
                                                                                                               top100_wiki,
                                                                                                               mrr_wiki))
            logger.info(
                "Conll, Epoch - {} Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(epoch,
                                                                                                           top1_conll,
                                                                                                           top10_conll,
                                                                                                           top100_conll,
                                                                                                           mrr_conll))
            if mrr_conll > self.best_valid_metric:
                best_model = self.model
                self.best_valid_metric = mrr_conll

        save_checkpoint({
            'state_dict': best_model.state_dict(),
            'optimizer': self.optimizer.state_dict()}, True, filename=join(self.model_dir, 'final_model.ckpt'))
