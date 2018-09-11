# Main training file
import pickle
from os.path import join

import numpy as np
from sklearn.model_selection import ParameterGrid

from src.train.combined import parse_args, setup
from src.utils.utils import get_model
from src.train.trainer import Trainer

np.warnings.filterwarnings('ignore')


def grid_search():
    param_grid = {'norm_final': [True, False],
                  'measure': ['ip', 'l2'],
                  'num_candidates': [16, 32, 64]
                  }
    result_dict = {}

    for param_dict in list(ParameterGrid(param_grid)):
        for k, v in param_dict.items():
            assert k in args.__dict__
            args.__dict__[k] = v
        logger.info("GRID SEARCH PARAMS : {}".format(param_dict))
        result_key = tuple(param_dict.items())
        result_dict[result_key] = {'Wikipedia': [],
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
        result_dict[result_key]['Wikipedia'].append((top1_wiki, top10_wiki, top100_wiki, mrr_wiki))
        result_dict[result_key]['Conll'].append((top1_conll, top10_conll, top100_conll, mrr_conll))
        logger.info('Dev Validation')
        logger.info("Wikipedia, Untrained Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(top1_wiki, top10_wiki, top100_wiki, mrr_wiki))
        logger.info("Conll, Untrained Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(top1_conll, top10_conll, top100_conll, mrr_conll))

        trainer = Trainer(loader=train_loader,
                          args=args,
                          validator=validator,
                          model=model,
                          model_dir=model_dir,
                          model_type='combined',
                          result_dict=result_dict,
                          result_key=result_key)
        logger.info("Starting Training")
        trainer.train()
        logger.info("Finished Training")
        print(result_dict)

    return result_dict


if __name__ == '__main__':
    args, logger, model_dir = parse_args()
    train_loader, validator, yamada_model, ent_embs, word_embs, gram_embs = setup(args, logger)
    result_dict = grid_search()
    with open(join(model_dir, 'grid_search_results.pickle'), 'wb') as f:
        pickle.dump(result_dict, f)
