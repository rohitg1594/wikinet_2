import sys
from logging import getLogger

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = getLogger(__name__)


class MuliOptim(object):

    def __init__(self, args=None, model=None):

        self.emb_flag = False
        self.other_flag = False

        optimizer_type = self.get_optim(optim=args.embs_optim)
        embs_param = self.get_embs_param(model)
        if embs_param:
            self.emb_flag = True
            if args.sparse:
                self.emb_optimizer = optimizer_type(embs_param, lr=args.lr)
            else:
                self.emb_optimizer = optimizer_type(embs_param, lr=args.lr, weight_decay=args.wd)
            self.emb_scheduler = ReduceLROnPlateau(self.emb_optimizer, mode='max', verbose=True, patience=5)

        optimizer_type = self.get_optim(optim=args.other_optim)
        other_param = self.get_other_param(model)
        if other_param:
            self.other_flag = True
            self.other_optimizer = optimizer_type(other_param, lr=args.lr, weight_decay=args.wd)
            self.other_scheduler = ReduceLROnPlateau(self.other_optimizer, mode='max', verbose=True, patience=5)

    def zero_grad(self):
        if self.emb_flag:
            self.emb_optimizer.zero_grad()
        if self.other_flag:
            self.other_optimizer.zero_grad()

    def optim_step(self):
        if self.emb_flag:
            self.emb_optimizer.step()
        if self.other_flag:
            self.other_optimizer.step()

    def scheduler_step(self, metric):
        if self.emb_flag:
            self.emb_scheduler.step(metric)
        if self.other_flag:
            self.other_scheduler.step(metric)

    def get_state_dict(self):
        state_dict = {}
        if self.emb_flag:
            state_dict['emb_optimizer'] = self.emb_optimizer.state_dict()
        if self.other_flag:
            state_dict['other_optimizer'] = self.other_optimizer.state_dict()

        return state_dict

    @staticmethod
    def get_embs_param(model):
        out = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                if 'embs' in n:
                    out.append(p)
                    print(f'Adding {n} to embs optimizer')
        return out

    @staticmethod
    def get_other_param(model):
        out = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                if 'embs' not in n:
                    out.append(p)
                    print(f'Adding {n} to other optimizer')
        return out

    @staticmethod
    def get_optim(optim=None):
        if optim == 'adagrad':
            optimizer = torch.optim.Adagrad
        elif optim == 'adam':
            optimizer = torch.optim.Adam
        elif optim == 'rmsprop':
            optimizer = torch.optim.RMSprop
        elif optim == 'sparseadam':
            optimizer = torch.optim.SparseAdam
        else:
            logger.error("Optimizer {} not recognized, choose between adam, adagrad, rmsprop, sparseadam".format(optim))
            sys.exit(1)

        return optimizer
