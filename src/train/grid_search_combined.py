# Main training file
import pickle
from os.path import join
import gc

import numpy as np
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import GridSearchCV

import pandas as pd

from src.train.combined import parse_args, setup
from src.utils.utils import get_model
from src.train.trainer import Trainer
from src.utils.dictionary import Dictionary  # needed because of autoencoder

np.warnings.filterwarnings('ignore')
DATA_TYPES = ['wiki', 'conll', 'msnbc', 'ace2004']


def grid_search(**kwargs):
    param_grid = {
                  'lr': [5e-2, 1e-3, 5e-3],
                  'wd': [1e-7, 1e-6, 1e-5],
                  'dp': [1e-1, 2e-1],
                  }
    results = {}
    pd_results = list()
    args = kwargs['args']
    logger = kwargs['logger']
    train_loader = kwargs['train_loader']
    validator = kwargs['validator']

    for param_dict in list(ParameterSampler(param_grid, n_iter=20)):
        for k, v in param_dict.items():
            assert k in args.__dict__
            args.__dict__[k] = v

        logger.info("GRID SEARCH PARAMS : {}".format(param_dict))
        result_key = tuple(param_dict.items())
        results[result_key] = {data_type: [] for data_type in DATA_TYPES}

        setup_dict['args'] = args

        # Model
        model = get_model(**setup_dict)

        logger.info("Validating untrained model.....")
        result = validator.validate(model=model, error=args.error)
        for data_type in DATA_TYPES:
            res_str = ""
            for k, v in result[data_type].items():
                res_str += k.upper() + ': {:.3},'.format(v)
            logger.info(f"{data_type}: Untrained, {res_str[:-1]}")
            results[result_key][data_type].append((tuple(result[data_type].values())))
        logger.info("Done validating.")

        trainer = Trainer(loader=train_loader,
                          args=args,
                          validator=validator,
                          model=model,
                          model_dir=Model_dir,
                          model_type='combined',
                          result_dict=results,
                          result_key=result_key)
        logger.info("Starting Training")
        best_results = trainer.train()
        logger.info("Finished Training")

        pd_results.append({**param_dict, **best_results})
        df = pd.DataFrame(pd_results)
        df.to_csv(join(Model_dir, 'hyper_df.csv'))

        with open(join(Model_dir, 'grid_search_results.pickle'), 'wb') as f:
            pickle.dump(results, f)

        del model, trainer
        gc.collect()

    return results, pd_results


if __name__ == '__main__':

    Args, Logger, Model_dir = parse_args()
    setup_dict = setup(args=Args, logger=Logger)
    setup_dict['logger'] = Logger
    setup_dict['args'] = Args
    setup_dict['model_dir'] = Model_dir

    result_dict, pd_dict = grid_search()
    df = pd.DataFrame(pd_dict)
    df.to_csv(join(Model_dir, 'hyper_df.csv'))
