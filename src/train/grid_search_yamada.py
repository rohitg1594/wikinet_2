# Main training file
import pickle
from os.path import join
import gc

import numpy as np
from sklearn.model_selection import ParameterGrid

import torch

from src.train.yamada import parse_args, setup, get_model
from src.train.trainer import Trainer

np.warnings.filterwarnings('ignore')


def grid_search():
    param_grid = {'dp': [0.1, 0.2, 0.3],
                  'hidden_size': [1000, 2000],
                  'lr': [0.01, 0.005],
                  'wd': [0.001, 0.0001],
                  }
    results = {}

    for param_dict in list(ParameterGrid(param_grid)):
        for k, v in param_dict.items():
            assert k in args.__dict__
            args.__dict__[k] = v
        logger.info("GRID SEARCH PARAMS : {}".format(param_dict))
        result_key = tuple(param_dict.items())
        results[result_key] = []

        model = get_model(args, yamada_model, logger)

        logger.info("Starting validation for untrained model.")
        correct, mentions = validator.validate(model)
        perc = correct / mentions * 100
        results[result_key].append(perc)
        logger.info('Untrained, Correct : {}, Mention : {}, Percentage : {}'.format(correct, mentions, perc))

        trainer = Trainer(loader=train_loader,
                          args=args,
                          validator=validator,
                          model=model,
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
    train_loader, validator, yamada_model = setup(args, logger)
    result_dict = grid_search()
