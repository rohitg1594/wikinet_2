# This module implements a trainer to be used by combined.py
import logging
import sys
from os.path import join
import cProfile
import gc

import torch
from torch.autograd import Variable
from src.utils.utils import save_checkpoint
from src.utils.multi_optim import MuliOptim

logger = logging.getLogger()


class Trainer(object):

    def __init__(self, loader=None, args=None, model=None, validator=None, model_type=None,
                 grid_results_dict=None, result_key=None, profile=False):
        self.loader = loader
        self.args = args
        self.model = model
        self.model_type = model_type
        self.num_epochs = self.args.num_epochs
        self.model_dir = self.args.model_dir
        self.min_delta = 1e-03
        self.patience = self.args.patience
        self.grid_results_dict = grid_results_dict
        self.result_key = result_key
        self.profile = profile
        self.data_types = ['conll', 'wiki', 'msnbc', 'ace2004']
        self.batch_size = args.batch_size
        self.validator = validator

        # Optimizer and scheduler
        self.optimizer = MuliOptim(args=self.args, model=self.model)

    def _get_next_batch(self, data_dict):
        skip_keys = ['ent_strs', 'cand_strs', 'not_in_cand', 'label']
        for k, v in data_dict.items():
            print(f'key: {k}, value type : {type(v)}')
            try:
                if k not in skip_keys:
                    data_dict[k] = Variable(v)
            except Exception as e:
                print(f'Exception : {e}, key - {k}, Value - {v}')
        if 'label' in data_dict:
            labels = Variable(data_dict['label'].type(torch.LongTensor), requires_grad=False)
        else:
            labels = Variable(torch.zeros(v.shape[0]).type(torch.LongTensor), requires_grad=False)

        for k in skip_keys:
            if k in data_dict:
                data_dict.pop(k)

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
        self.optimizer.optim_step()

        return loss.item()

    def combined_validate(self, epoch):
        results = self.validator.validate(model=self.model, error=self.args.error)

        for data_type in self.data_types:
            res_str = ""
            for k, v in results[data_type].items():
                if isinstance(k, int):
                    res_str += f'TOP{k}'
                else:
                    res_str += k.upper()
                res_str += ': {:.3},'.format(v)
            logger.info(f"{data_type.upper()}: Epoch {epoch}," + res_str[:-1])
            if self.grid_results_dict is not None:
                self.grid_results_dict[self.result_key][data_type].append((tuple(results[data_type].values())))

        return results

    def yamada_validate(self, epoch):
        results = {}
        for data_type in self.data_types:
            total_mentions, not_in_cand, correct, cor_adjust = self.validator[data_type].validate(self.model)
            logger.info(
                f'Total mentions : {total_mentions}, Not in Cand: {not_in_cand}, Correct : {correct}, Correct Adjust: {cor_adjust}')
            cand_coverage = (1 - not_in_cand / total_mentions) * 100
            acc = (correct - cor_adjust) / total_mentions * 100
            logger.info(f'Epoch : {epoch}, {data_type}, Cand Coverage - {cand_coverage}, Acc- {acc}')

            results[data_type] = acc
            if self.result_key is not None:
                self.grid_results_dict[self.result_key][data_type].append(acc)

        return results

    def _profile(self):
        logger.info('Starting profiling of dataloader.....')
        pr = cProfile.Profile()
        pr.enable()

        for batch_idx, _ in enumerate(self.loader, 0):
            print(batch_idx)
            pass

        pr.disable()
        pr.dump_stats(join(self.args.data_path, 'stats.prof'))
        pr.print_stats(sort='time')

        sys.exit()

    def train(self):
        training_losses = []

        wait = 0
        num_batches = len(self.loader)
        tenth_batch = num_batches // 10
        TOP_VALID = 1

        valid_func = self.combined_validate if self.model_type == 'combined' else self.yamada_validate

        logger.info("Validating untrained model.....")
        best_model = self.model
        full_results = valid_func('Untrained')
        best_results = {data_type: result[TOP_VALID] if isinstance(result, dict) else result for data_type, result in full_results.items()}
        best_valid_metric = best_results['conll']
        logger.info("Done validating.")

        if self.profile:
            self._profile()

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
                    **self.optimizer.get_state_dict()},
                    filename=join(self.model_dir, '{}.ckpt'.format(epoch)))

            results = valid_func(epoch)
            valid_metric = results['conll'][TOP_VALID] if isinstance(results['conll'], dict) else results['conll']
            for data_type, result in results.items():
                top1 = result[1] if isinstance(result, dict) else result
                if top1 > best_results[data_type]:
                    best_results[data_type] = top1

            self.optimizer.scheduler_step(valid_metric)

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
            **self.optimizer.get_state_dict()}, filename=join(self.model_dir, 'best_model.ckpt'))

        return best_results
