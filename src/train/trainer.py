# This module implements a trainer to be used by combined.py
import logging
import sys
from os.path import join
import cProfile
import gc

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.utils import save_checkpoint, get_optim

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

        optimizer_type = get_optim(optim=self.args.optim)
        if self.args.optim == 'sparseadam':
            self.optimizer = optimizer_type(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            self.optimizer = optimizer_type(filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=args.lr,
                                            weight_decay=args.wd)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', verbose=True, patience=5)
        self.loader_index = 0

    def _get_next_batch(self, data_dict):
        for k, v in data_dict.items():
            data_dict[k] = Variable(v)

        labels = Variable(torch.zeros(v.shape[0]).type(torch.LongTensor), requires_grad=False)

        if self.args.use_cuda:
            device = self.args.device if isinstance(self.args.device, int) else self.args.device[0]
            for k, v in data_dict.items():
                data_dict[k] = v.cuda(device)
            labels = labels.cuda(device)

        return data_dict, labels

    def step(self, data):
        data, labels = self._get_next_batch(data)
        scores, _, _ = self.model(data)
        loss = self.model.loss(scores, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

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
            total_mentions, not_in_cand, correct = self.validator[data_type].validate(self.model)
            logger.info(
                f'Total mentions : {total_mentions}, Not in Cand: {not_in_cand}, Correct : {correct}')
            cand_coverage = (1 - not_in_cand / total_mentions) * 100
            acc = (correct - not_in_cand) / total_mentions * 100
            logger.info(f'Epoch : {epoch}, {data_type}, Cand Coverage - {cand_coverage}, Acc- {acc}')

            results[data_type] = acc
            if self.result_key is not None:
                self.result_dict[self.result_key][data_type].append(acc)

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
            self.model.train()
            assert self.model.train()
            for batch_no, data in enumerate(self.loader, 0):
                if batch_no % tenth_batch == 0:
                    logger.info("Now on batch - {}".format(batch_no))
                loss = self.step(data)
                training_losses.append(loss)

            # Free memory allocated by data
            data = {k: data[k].cpu() for k in data.keys()}
            del data
            torch.cuda.empty_cache()
            gc.collect()

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
