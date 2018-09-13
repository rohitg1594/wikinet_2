# Main training file
import pickle
from os.path import join
import gc

import numpy as np
from sklearn.model_selection import ParameterGrid

import torch

from src.train.yamada import parse_args, setup
from src.train.trainer import Trainer

np.warnings.filterwarnings('ignore')


def grid_search(model_in):
    param_grid = {'dp': [0, 0.1, 0.2, 0.3],
                  'hidden_size': [250, 500, 1000, 2000],
                  'lr': [0.01, 0.005, 0.001],
                  'wd': [0.001, 0.0001, 0.0005]
                  }
    results = {}

    for param_dict in list(ParameterGrid(param_grid)):
        for k, v in param_dict.items():
            assert k in args.__dict__
            args.__dict__[k] = v
        logger.info("GRID SEARCH PARAMS : {}".format(param_dict))
        result_key = tuple(param_dict.items())
        results[result_key] = []

        logger.info("Starting validation for untrained model.")
        correct, mentions = validator.validate(model_in)
        perc = correct / mentions * 100
        logger.info('Untrained, Correct : {}, Mention : {}, Percentage : {}'.format(correct, mentions, perc))

        trainer = Trainer(loader=train_loader,
                          args=args,
                          validator=validator,
                          model=model_in,
                          model_dir=model_dir,
                          model_type='yamada',
                          result_dict=results,
                          result_key=result_key)
        logger.info("Starting Training")
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

    args, logger, model_dir = parse_args()
    train_loader, validator, yamada_model, model = setup(args, logger)
    result_dict = grid_search(model)
