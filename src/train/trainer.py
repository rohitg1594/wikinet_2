# This module implements a trainer to be used by combined.py
import logging
import sys
from os.path import join
import cProfile

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.data import save_checkpoint

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
        self.batch_size = args.batch_size
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
        for i in range(len(data)):
            data[i] = Variable(data[i])

        labels = Variable(torch.zeros(data[0].shape[0]).type(torch.LongTensor), requires_grad=False)

        if self.args.use_cuda:
            if isinstance(self.args.device, int):
                for i in range(len(data)):
                    data[i] = data[i].cuda(self.args.device)
                labels = labels.cuda(self.args.device)
            else:
                labels = labels.cuda(self.args.device[0])

        return tuple(data), labels

    def step(self, data):
        data, labels = self._get_next_batch(data)
        scores, _, _ = self.model(data)
        loss = self.model.loss(scores, labels)

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
        results = {}
        for data_type in self.data_types:
            correct, mentions, cand_correction= self.validator[data_type].validate(self.model)
            acc_not_corrected = correct / mentions * 100
            acc_corrected = (correct - cand_correction) / mentions * 100
            logger.info(f'Epoch - {epoch}, {data_type} - {acc_not_corrected}, {acc_corrected}')

            if self.result_key is not None:
                self.result_dict[self.result_key][data_type].append(acc_corrected)

        return results['conll']

    def train(self):
        training_losses = []
        best_model = self.model
        wait = 0
        best_valid_metric = 0
        best_results = {k: 0 for k in self.data_types}
        num_batches = len(self.loader)
        tenth_batch = num_batches // 10

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
                if batch_no % tenth_batch == 0:
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

        return best_results
