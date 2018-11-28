# Main training file
import pickle
from os.path import join
import gc

import numpy as np
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import ParameterGrid

import pandas as pd

from src.train.combined import parse_args, setup
from src.utils.utils import get_model, pickle_dump
from src.train.trainer import Trainer
from src.utils.dictionary import Dictionary  # needed because of autoencoder

np.warnings.filterwarnings('ignore')
DATA_TYPES = ['wiki', 'conll', 'msnbc', 'ace2004']


def grid_search(**kwargs):
    param_grid = {
                  'lr': [1e-4, 5e-4, 1e-3, 5e-3],
                  'wd': [1e-8, 1e-7, 1e-6],
                  'dp': [1e-1, 2e-1, 0],
                  'init_linear': ['kaiming_uniform_', 'kaiming_normal_', 'xavier_uniform_', 'xavier_normal_'],
                  'num_linear': [1, 2, 3, 4],
                  }

    grid_results_dict = {}
    pd_results = list()
    args = kwargs['args']
    logger = kwargs['logger']
    train_loader = kwargs['train_loader']
    validator = kwargs['validator']

    for param_dict in list(ParameterSampler(param_grid, 20)):
        for k, v in param_dict.items():
            assert k in args.__dict__
            args.__dict__[k] = v

        # Setup
        logger.info("GRID SEARCH PARAMS : {}".format(param_dict))
        result_key = tuple(param_dict.items())
        grid_results_dict[result_key] = {data_type: [] for data_type in DATA_TYPES}
        setup_dict['args'] = args

        # Model
        model = get_model(**setup_dict)

        # Train
        trainer = Trainer(loader=train_loader,
                          args=args,
                          validator=validator,
                          model=model,
                          model_type='combined',
                          grid_results_dict=grid_results_dict,
                          result_key=result_key)
        logger.info("Starting Training")
        best_results = trainer.train()
        logger.info("Finished Training")

        # Results
        pd_results.append({**param_dict, **best_results})
        df = pd.DataFrame(pd_results)
        df.to_csv(join(args.model_dir, 'hyper_df.csv'))

        pickle_dump(grid_results_dict, join(args.model_dir, 'grid_search_results.pickle'))

        del model, trainer
        gc.collect()

    return pd_results


if __name__ == '__main__':

    Args, Logger= parse_args()
    setup_dict = setup(args=Args, logger=Logger)
    setup_dict['logger'] = Logger
    setup_dict['args'] = Args

    pd_dict = grid_search(**setup_dict)
    df = pd.DataFrame(pd_dict)
    df.to_csv(join(Args.model_dir, 'hyper_df.csv'))
