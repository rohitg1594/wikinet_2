# Main training file
import pickle
from os.path import join
import gc
from collections import defaultdict

import numpy as np
from sklearn.model_selection import ParameterGrid

import torch

from src.train.yamada import parse_args, setup, get_model
from src.train.trainer import Trainer
DATA_TYPES = ['wiki', 'conll', 'ace2004', 'msnbc']

np.warnings.filterwarnings('ignore')


def grid_search(yamada_model=None,
                logger=None,
                validators=None,
                model_dir=None,
                train_loader=None,
                args=None):
    param_grid = {'dp': [0.1, 0.2, 0.3],
                  'hidden_size': [1000, 2000],
                  'lr': [0.01, 0.005],
                  'wd': [0.001, 0.0001],
                  'num_docs': [10, 100, 1000, 10000]
                  }
    results = defaultdict(dict)

    for param_dict in list(ParameterGrid(param_grid)):
        for k, v in param_dict.items():
            assert k in args.__dict__
            args.__dict__[k] = v

        model = get_model(args, yamada_model, logger)

        logger.info("GRID SEARCH PARAMS : {}".format(param_dict))
        result_key = tuple(param_dict.items())
        logger.info("Starting validation for untrained model.....")
        for data_type in DATA_TYPES:
            correct, mentions = validators[data_type].validate(model)
            res = correct / mentions * 100
            results[result_key][data_type] = [res]
            logger.info(f'Untrained, {data_type} - {res}')

        trainer = Trainer(loader=train_loader,
                          args=args,
                          validator=validators,
                          model=model,
                          model_dir=model_dir,
                          model_type='yamada',
                          result_dict=results,
                          result_key=result_key)
        logger.info("Starting Training.....")
        print()
        trainer.train()
        logger.info("Finished Training")

        for k, v in results.items():
            print(k)
            print(v)

        with open(join(model_dir, 'grid_search_results.pickle'), 'wb') as f:
            pickle.dump(results, f)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return results


if __name__ == '__main__':
    Args, Logger, Model_dir = parse_args()
    Train_loader, Validators, Yamada_model = setup(Args, Logger)
    result_dict = grid_search(yamada_model=Yamada_model,
                              model_dir=Model_dir,
                              validators=Validators,
                              train_loader=Train_loader,
                              logger=Logger,
                              args=Args)
