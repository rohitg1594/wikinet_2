# Main training file
import pickle
from os.path import join
import gc

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler

import torch
import pandas as pd

from src.train.combined import parse_args, setup
from src.utils.utils import get_model
from src.train.trainer import Trainer

np.warnings.filterwarnings('ignore')
DATA_TYPES = ['wiki', 'conll', 'msnbc', 'ace2004']


def grid_search():
    param_grid = {
                  'lr': [5e-2, 1e-2, 5e-3, 1e-3],
                  'wd': [1e-5, 1e-6, 1e-7],
                  'ent_mention_dim': [128],
                  'init_stdv': [1e-2, 5e-2],
                  'combined_linear': [False],
                  'dp': [0, 1e-1, 2e-1, 3e-1],
                  }
    results = {}
    pd_results = list()

    for param_dict in list(ParameterSampler(param_grid, n_iter=20)):
        param_dict['context_word_dim'] = param_dict['ent_mention_dim'] // 2
        param_dict['mention_word_dim'] = param_dict['ent_mention_dim'] // 2
        for k, v in param_dict.items():
            assert k in args.__dict__
            args.__dict__[k] = v

        logger.info("GRID SEARCH PARAMS : {}".format(param_dict))
        result_key = tuple(param_dict.items())
        results[result_key] = {data_type: [] for data_type in DATA_TYPES}
        # Model
        model = get_model(args,
                          yamada_model=yamada_model,
                          ent_embs=ent_embs,
                          word_embs=word_embs,
                          gram_embs=gram_embs,
                          init=args.init_mention)

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
                          model_dir=model_dir,
                          model_type='combined',
                          result_dict=results,
                          result_key=result_key)
        logger.info("Starting Training")
        best_model, best_results = trainer.train()
        logger.info("Finished Training")

        pd_results.append({**param_dict, **best_results})
        print('PD RESULTS: {}'.format(pd_results))
        df = pd.DataFrame(pd_results)
        df.to_csv(join(model_dir, 'hyper_df.csv'))

        with open(join(model_dir, 'grid_search_results.pickle'), 'wb') as f:
            pickle.dump(results, f)

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    return results, pd_results


if __name__ == '__main__':

    args, logger, model_dir = parse_args()
    train_loader, validator, yamada_model, ent_embs, word_embs, gram_embs = setup(args, logger)
    result_dict, pd_dict = grid_search()
    df = pd.DataFrame(pd_dict)
    df.to_csv(join(model_dir, 'hyper_df.csv'))
