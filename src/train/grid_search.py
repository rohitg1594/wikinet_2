# Main training file
import pickle
from os.path import join
import gc
import sys

import numpy as np
from sklearn.model_selection import ParameterGrid

import torch

np.warnings.filterwarnings('ignore')


def grid_search_combined():
    param_grid = {'num_candidates': [32, 64, 128, 256],
                  'mention_word_dim': [64, 128],
                  'lr': [0.001, 0.005, 0.0001],
                  'wd': [0.0001, 0.0005]
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
        # Model
        model = get_model(args,
                          yamada_model=yamada_model,
                          ent_embs=ent_embs,
                          word_embs=word_embs,
                          gram_embs=gram_embs,
                          init=args.init_mention)

        logger.info("Starting validation for untrained model.")
        top1_wiki, top10_wiki, top100_wiki, mrr_wiki, top1_conll, top10_conll, top100_conll, mrr_conll = validator.validate(model=model)
        results[result_key]['Wikipedia'].append((top1_wiki, top10_wiki, top100_wiki, mrr_wiki))
        results[result_key]['Conll'].append((top1_conll, top10_conll, top100_conll, mrr_conll))
        logger.info('Dev Validation')
        logger.info("Wikipedia, Untrained Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(top1_wiki, top10_wiki, top100_wiki, mrr_wiki))
        logger.info("Conll, Untrained Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(top1_conll, top10_conll, top100_conll, mrr_conll))

        trainer = Trainer(loader=train_loader,
                          args=args,
                          validator=validator,
                          model=model,
                          model_dir=model_dir,
                          model_type='combined',
                          result_dict=results,
                          result_key=result_key)
        logger.info("Starting Training")
        trainer.train()
        logger.info("Finished Training")

        for k, v in results.items():
            print(k)
            print('WIKI')
            print(v['Wikipedia'])
            print('CONLL')
            print(v['Conll'])
        with open(join(model_dir, 'grid_search_results.pickle'), 'wb') as f:
            pickle.dump(results, f)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return results


def grid_search_yamada():
    param_grid = {'dp': [0, 0.1, 0.2, 0.3],
                  'hidden_size': [64, 128],
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
        correct, mentions = validator.validate(model)
        perc = correct / mentions * 100
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
    if sys.argv[1] == 'combined':
        from src.train.combined import parse_args, setup
        from src.utils.utils import get_model
        from src.train.trainer import Trainer
        args, logger, model_dir = parse_args()
        train_loader, validator, yamada_model, ent_embs, word_embs, gram_embs = setup(args, logger)
        result_dict = grid_search_combined()

    else:
        from src.train.yamada import parse_args, setup
        from src.train.trainer import Trainer

        args, logger, model_dir = parse_args()
        train_loader, validator, yamada_model, model = setup(args, logger)
        result_dict = grid_search_combined()
