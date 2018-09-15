# Main training file
import pickle
from os.path import join
import gc

import numpy as np
from sklearn.model_selection import ParameterGrid

import torch

from src.train.yamada import parse_args, setup, get_model
from src.train.trainer import Trainer
from src.utils.utils import yamada_validate_wrap

np.warnings.filterwarnings('ignore')


def grid_search(yamada_model=None,
                logger=None,
                conll_validator=None,
                wiki_validator=None,
                model_dir=None,
                train_loader=None,
                args=None):
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
        results[result_key] = {'Wikipedia': [],
                               'Conll': []}

        model = get_model(args, yamada_model, logger)

        logger.info("Starting validation for untrained model.....")
        conll_perc, wiki_perc = yamada_validate_wrap(conll_validator=conll_validator,
                                                     wiki_validator=wiki_validator,
                                                     model=model)
        results[result_key]['Conll'].append(conll_perc)
        results[result_key]['Wikipedia'].append(wiki_perc)
        logger.info('Untrained, Conll - {}'.format(conll_perc))
        logger.info('Untrained, Wiki - {}'.format(wiki_perc))

        trainer = Trainer(loader=train_loader,
                          args=args,
                          validator=(conll_validator, wiki_validator),
                          model=model,
                          model_dir=model_dir,
                          model_type='yamada',
                          result_dict=results,
                          result_key=result_key)
        logger.info("Starting Training:")
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
    Train_loader, Conll_validator, Wiki_validator, Yamada_model = setup(Args, Logger)
    result_dict = grid_search(yamada_model=Yamada_model,
                              conll_validator=Conll_validator,
                              wiki_validator=Wiki_validator,
                              model_dir=Model_dir,
                              train_loader=Train_loader,
                              logger=Logger,
                              args=Args)
