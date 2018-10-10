#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime
import inspect
import os
import pickle

import numpy
import pandas
from ast import literal_eval
from collections import OrderedDict
from copy import deepcopy
from itertools import product

from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid, ParameterSampler

from deepreg.train import parser, main, Models
from deepreg.datasets import *

parser.add_argument('--skip',  default=0, type=int, help='Skip this many experiments')
parser.add_argument('--train_iter',  default=20, type=int, help='Number of training parameter samples')
parser.add_argument('--model_iter',  default=5, type=int, help='Number of model parameter samples')

def isnumeric(n):
    try:
        float(n)
        if isinstance(n, bool):
            return False
        return True
    except:
        return False

if __name__ == '__main__':

    args = parser.parse_args()
    if isinstance(args.model_config, str):
        args.model_config = literal_eval(args.model_config)
    orig_args = deepcopy(args)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    orig_args.results_dir = os.path.join(args.results_dir, 'hyper', config_name, time_stamp)

    model_param_grid = {
        'mention_word_dim': [64, 128, 256],
        'context_word_dim': [128, 256],
        'ent_mention_dim': [128, 256],
    }

    train_param_grid = {
        'lr': [5e-2, 1e-2, 5e-3, 1e-3, ],
        'weight_decay': [1e-5, 1e-6, 1e-7, ],
        'init_stdv': [1e-2, 1e-3],
    }

    model_param_grid = OrderedDict({k:v for k,v in model_param_grid.items() if k in args.model_config})
    train_param_grid = OrderedDict(train_param_grid)

    print(model_param_grid)

    results = list()
    header = None

    val_loss_col = ['val_loss']

    model_param_cols = ['{}'.format(k) for k, v in model_param_grid.items()]
    train_param_cols = ['{}'.format(k) for k, v in train_param_grid.items()]

    numeric_cols = [k for k in train_param_cols if isnumeric(train_param_grid[k][0])] + \
                   [k for k in model_param_cols if isnumeric(model_param_grid[k][0])]

    categorial_cols = [k for k in train_param_cols if not isnumeric(train_param_grid[k][0])] + \
                   [k for k in model_param_cols if not isnumeric(model_param_grid[k][0])]

    hyper_experiment_nr = 0
    for train_cell in ParameterSampler(train_param_grid, n_iter=args.train_iter):
        for model_cell in ParameterSampler(model_param_grid, n_iter=args.model_iter):

            hyper_experiment_nr += 1
            if hyper_experiment_nr <= args.skip:
                continue
            args = deepcopy(orig_args)
            args.optimization_config = [{'epoch': 0, 'optimizer': train_cell['optimizer'], 'lr': train_cell['lr'], 'weight_decay': train_cell['weight_decay']}]

            for k,v in model_cell.items():
                if k in args.model_config:
                    args.model_config[k] = v

            # print('Batch size: {}'.format(args.batch_size))
            # print(hyper_experiment_nr, 'hyper#' + '#'.join(['{}^{}'.format(k,v) for k,v in model_cell.items()] + ['{}_{}'.format(k,v) for k,v in train_cell.items()]))
            val_loss = main(args, 'hyper#' + '#'.join(['{}_{}'.format(k,v) for k,v in model_cell.items()] + ['{}_{}'.format(k,v) for k,v in train_cell.items()]))
            # val_loss = numpy.random.randn()

            results.append([train_cell[k] for k in train_param_cols] + [model_cell[k] for k in model_param_cols] + [val_loss])

            df = pandas.DataFrame(results, columns=train_param_cols + model_param_cols + val_loss_col)
            df = pandas.get_dummies(df, columns=categorial_cols)
            columns_list_without_loss = list(df.columns.tolist())
            columns_list_without_loss.remove(val_loss_col[0])
            columns_list_with_loss = columns_list_without_loss + val_loss_col
            df[columns_list_with_loss].to_csv(os.path.join(args.results_dir, '{}-{}.{}'.format(config_name, time_stamp, 'csv')))
            linreg = Ridge(normalize=True)
            linreg.fit(df.drop(columns=val_loss_col), df[val_loss_col])
            linreg_coefs = list(zip(columns_list_without_loss, *linreg.coef_))
            pandas.DataFrame(linreg_coefs).to_csv(os.path.join(args.results_dir, '{}-{}-{}.{}'.format(config_name, time_stamp, 'lincoefs', 'csv')))
            df[columns_list_with_loss].corr().to_csv(os.path.join(args.results_dir, '{}-{}-{}.{}'.format(config_name, time_stamp, 'corr', 'csv')))