# This module implements a trainer to be used by combined.py
import logging

import sys
from os.path import join

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.data import save_checkpoint
from src.utils.utils import yamada_validate_wrap

import cProfile

logger = logging.getLogger()


class Trainer(object):

    def __init__(self, loader=None, args=None, model=None, validator=None, model_dir=None, model_type=None,
                 result_dict=None, result_key=None, profile=False):
        self.loader = loader
        self.args = args
        self.model = model
        self.model_type = model_type
        self.num_epochs = self.args.num_epochs
        self.model_dir = model_dir
        self.min_delta = 1e-03
        self.patience = self.args.patience
        self.result_dict = result_dict
        self.result_key = result_key
        self.profile = profile
        self.data_types = ['wiki', 'conll', 'msnbc', 'ace2004']

        if isinstance(validator, tuple):
            self.conll_validator, self.wiki_validator = validator
        else:
            self.validator = validator

        if args.optim == 'adagrad':
            optimizer_type = torch.optim.Adagrad
        elif args.optim == 'adam':
            optimizer_type = torch.optim.Adam
        elif args.optim == 'rmsprop':
            optimizer_type = torch.optim.RMSprop
        else:
            logger.error("Optimizer {} not recognized, choose between adam, adagrad, rmsprop".format(args.optim))
            sys.exit(1)

        self.optimizer = optimizer_type(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=args.lr,
                                        weight_decay=args.wd)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', verbose=True, patience=5)
        self.loader_index = 0

    def _get_next_batch(self, data):
        data = list(data)
        ymask = data[0]
        b, e = ymask.shape
        data = data[1:]
        for i in range(len(data)):
            data[i] = Variable(data[i])

        ymask = ymask.view(b * e)
        ymask = Variable(ymask)
        labels = Variable(torch.zeros(b * e).type(torch.LongTensor), requires_grad=False)

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
        scores = self.model(data)
        if isinstance(scores, tuple) > 1:
            scores, labels = scores

        if self.args.loss_func == 'cosine':
            loss = self._cosine_loss(scores, ymask)
        elif self.args.loss_func == 'cross_entropy':
            loss = self._cross_entropy(scores, ymask, labels)
        else:
            logger.error("Loss function {} not recognized, choose one of cosine, cross_entropy")
            sys.exit(1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def combined_validate(self, epoch):
        results = self.validator.validate(model=self.model, error=self.args.error)

        for data_type in self.data_types:
            res_str = ""
            for k, v in results[data_type].items():
                res_str += k.upper() + ': {:.3},'.format(v)
            logger.info(f"{data_type}: Epoch {epoch}," + res_str[:-1])
            if self.result_dict is not None:
                self.result_dict[self.result_key][data_type].append((tuple(results[data_type].values())))

        return results

    def yamada_validate(self, epoch):
        conll_perc, wiki_perc = yamada_validate_wrap(conll_validator=self.conll_validator,
                                                     wiki_validator=self.wiki_validator,
                                                     model=self.model)
        logger.info('Epoch - {}, Conll - {}'.format(epoch, conll_perc))
        logger.info('Epoch - {}, Wiki - {}'.format(epoch, wiki_perc))
        if self.result_key is not None:
            self.result_dict[self.result_key]['Conll'].append(conll_perc)
            self.result_dict[self.result_key]['Wikipedia'].append(wiki_perc)

        return conll_perc

    def train(self):
        training_losses = []
        best_model = self.model
        wait = 0
        best_valid_metric = 0
        best_results = {k:0 for k in self.data_types}
        num_batches = len(self.loader)
        if num_batches > 1000:
            batch_verbose = True
        else:
            batch_verbose = False

        if self.profile:
            logger.info('Starting profiling of dataloader.....')
            pr = cProfile.Profile()
            pr.enable()

            for _, _ in enumerate(self.loader, 0):
                pass

            pr.disable()
            pr.dump_stats(join(self.args.data_path, 'stats.prof'))
            pr.print_stats(sort='time')

            sys.exit()

        for epoch in range(self.num_epochs):
            self.model = self.model.train()
            for batch_no, data in enumerate(self.loader, 0):
                if batch_no % 1000 == 0 and batch_verbose:
                    logger.info("Now on batch - {}".format(batch_no))
                loss = self.step(data)
                training_losses.append(loss)

            logger.info('Epoch - {}, Training Loss - {:.4}'.format(epoch, loss))
            if epoch % self.args.save_every == 0 and epoch != 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, filename=join(self.model_dir, '{}.ckpt'.format(epoch)))

            if self.model_type == 'combined':
                results = self.combined_validate(epoch)
                valid_metric = results['conll']['top1']
                for data_type, result in results.items():
                    top1 = result['top1']
                    if top1 > best_results[data_type]:
                        best_results[data_type] = top1
            elif self.model_type == 'yamada':
                valid_metric = self.yamada_validate(epoch)
            else:
                logger.error("Model {} not recognized, choose between combined, yamada".format(self.args.model_type))
                sys.exit(1)

            self.scheduler.step(valid_metric)

            if valid_metric > best_valid_metric:
                best_model = self.model
                best_valid_metric = valid_metric
                wait = 0
            else:
                if wait >= self.patience:
                    logger.info("Network not improving, breaking at epoch {}".format(epoch))
                    break
                wait += 1

        save_checkpoint({
            'state_dict': best_model.state_dict(),
            'optimizer': self.optimizer.state_dict()}, filename=join(self.model_dir, 'best_model.ckpt'))

        return best_model, best_results
