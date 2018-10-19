# Main training file
import pickle
from os.path import join
import gc
from collections import defaultdict

import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd

import torch

from src.train.yamada import parse_args, setup, get_model
from src.train.trainer import Trainer
from src.eval.yamada import YamadaValidator
DATA_TYPES = ['wiki', 'conll', 'ace2004', 'msnbc']

np.warnings.filterwarnings('ignore')


def grid_search(yamada_model=None,
                logger=None,
                datasets=None,
                model_dir=None,
                train_dataset=None,
                args=None):
    param_grid = {'dp': [0.1, 0.2, 0.3],
                  'hidden_size': [1000, 2000],
                  'lr': [1e-2, 5e-2, 1e-3],
                  'wd': [1e-3, 1e-4, 1e-5],
                  'num_docs': [100, 50, 10]
                  }
    results = defaultdict(dict)
    pd_results = list()

    for param_dict in list(ParameterGrid(param_grid)):
        for k, v in param_dict.items():
            assert k in args.__dict__
            args.__dict__[k] = v
        args.__dict__['batch_size'] = 10000 // param_dict['num_docs']

        model = get_model(args, yamada_model, logger)
        train_loader = train_dataset.get_loader(batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                drop_last=False)

        logger.info("GRID SEARCH PARAMS : {}".format(param_dict))
        print(f'Batch Size : {args.batch_size}')
        result_key = tuple(param_dict.items())

        logger.info("Starting validation for untrained model.....")
        validators = {}
        for data_type in DATA_TYPES:
            loader = datasets[data_type].get_loader(batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers,
                                                    drop_last=False)
            logger.info(f'Len loader {data_type} : {len(loader)}')
            validators[data_type] = YamadaValidator(loader=loader, args=args,
                                                    word_dict=yamada_model['word_dict'],
                                                    ent_dict=yamada_model['ent_dict'])
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
        best_results = trainer.train()
        logger.info("Finished Training")

        pd_results.append({**param_dict, **best_results})
        print('PD RESULTS: {}'.format(pd_results))
        df = pd.DataFrame(pd_results)
        df.to_csv(join(model_dir, 'hyper_df.csv'))

        for k, v in results.items():
            print(k)
            print(v)

        with open(join(model_dir, 'grid_search_results.pickle'), 'wb') as f:
            pickle.dump(results, f)

        del model, trainer, train_loader, loader, validators
        torch.cuda.empty_cache()
        gc.collect()

    return results, pd_results


if __name__ == '__main__':
    Args, Logger, Model_dir = parse_args()
    Train_dataset, Datasets, Yamada_model = setup(Args, Logger)
    result_dict, pd_dict = grid_search(yamada_model=Yamada_model,
                                       model_dir=Model_dir,
                                       train_dataset=Train_dataset,
                                       datasets=Datasets,
                                       logger=Logger,
                                       args=Args)
    df = pd.DataFrame(pd_dict)
    df.to_csv(join(Model_dir, 'hyper_df.csv'))
