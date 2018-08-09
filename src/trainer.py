# This module implements a trainer to be used by train.py
import logging

import sys

import torch

logger = logging.getLogger()


class Trainer(object):

    def __init__(self, loader, args, model):
        self.loader = loader
        self.args = args
        self.model = model

        if args.optim == 'adagrad':
            self.optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=args.lr,
                                            weight_decay=args.wd)
        elif args.optim == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=args.lr,
                                         weight_decay=args.wd)
        else:
            logger.error("Optimizer {} not recognized, choose between adam, adagrad".format(args.optim))
            sys.exit(1)

    def _get_next_batch(self):
        pass
