# Training file for original yamada model
import os
from os.path import join
from datetime import datetime
import sys

import numpy as np

from torch.nn import DataParallel

import configargparse

from src.utils.utils import str2bool
from src.utils.data import pickle_load, load_data
from src.conll.pershina import PershinaExamples
from src.dataloaders.yamada import YamadaDataset
from src.eval.yamada import YamadaValidator
from src.models.yamada.yamada_context import YamadaContext
from src.models.yamada.yamada_context_stats import YamadaContextStats
from src.models.yamada.yamada_context_stats_string import YamadaContextStatsString
from src.models.yamada.yamada_context_string import YamadaContextString
from src.logger import get_logger
from src.train.trainer import Trainer

np.warnings.filterwarnings('ignore')


def parse_args():
    parser = configargparse.ArgumentParser(description='Training Wikinet 2',
                                           formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    # General
    general = parser.add_argument_group('General Settings.')
    general.add_argument('--my-config', required=True, is_config_file=True, help='config file path')
    general.add_argument('--seed', type=int, default=-1, help="Initialization seed")
    general.add_argument('--exp_name', type=str, default="debug", help="Experiment name")
    general.add_argument("--debug", type=str2bool, default=True, help="whether to debug")

    # Data
    data = parser.add_argument_group('Data Settings.')
    data.add_argument('--data_path', type=str, help='location of data dir')
    data.add_argument('--yamada_model', type=str, help='name of yamada model')
    data.add_argument('--data_type', type=str, choices=['conll', 'wiki'], help='whether to train with conll or wiki')
    data.add_argument('--proto_data', type=str2bool, help='whether to use prototype data')
    data.add_argument('--num_shards', type=int, help='number of shards of training file')
    data.add_argument('--train_size', type=int, help='number of training abstracts')

    # Max Padding
    padding = parser.add_argument_group('Max Padding for batch.')
    padding.add_argument('--max_context_size', type=int, help='max number of context')
    padding.add_argument('--max_ent_size', type=int, help='max number of entities considered in abstract')
    padding.add_argument('--ignore_init', type=str2bool, help='whether to ignore first five tokens of context')

    # Model Type
    model_selection = parser.add_argument_group('Type of model to train.')
    model_selection.add_argument('--include_string', type=str2bool,
                                 help='whether to include string information in yamada model')
    model_selection.add_argument('--include_stats', type=str2bool,
                                 help='whether to include stats information in yamada model')

    # Model params
    model_params = parser.add_argument_group("Parameters for chosen model.")
    model_params.add_argument('--dp', type=float, help='drop out')
    model_params.add_argument('--hidden_size', type=int, help='size of hidden layer in yamada model')

    # Candidate Generation
    candidate = parser.add_argument_group('Candidate generation.')
    candidate.add_argument('--cand_type', choices=['necounts', 'pershina'], help='whether to use pershina candidates')
    candidate.add_argument('--cand_gen_rand', type=str2bool, help='whether to generate random candidates')
    candidate.add_argument("--num_candidates", type=int, default=32, help="Number of candidates")

    # Training
    train = parser.add_argument_group("Training parameters.")
    train.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    train.add_argument("--save_every", type=int, default=5, help="how often to checkpoint")
    train.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    train.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train.add_argument("--num_workers", type=int, default=4, help="number of workers for data loader")
    train.add_argument('--lr', type=float, help='learning rate')
    train.add_argument('--wd', type=float, help='weight decay')
    train.add_argument('--optim', type=str, choices=['adagrad', 'adam', 'rmsprop'], help='optimizer')
    train.add_argument('--sparse', type=str2bool, help='sparse gradients')

    # Loss
    loss = parser.add_argument_group('Type of loss.')
    loss.add_argument('--loss_func', type=str, default='cross_entropy', choices=['cross_entropy', 'cosine'],
                      help='loss function')
    loss.add_argument('--margin', type=float, help='margin of hinge loss')

    # cuda
    parser.add_argument("--device", type=str, help="cuda device")
    parser.add_argument("--use_cuda", type=str2bool, help="use gpu or not")

    args = parser.parse_args()
    logger = get_logger(args)

    if args.wd > 0:
        assert not args.sparse

    if args.use_cuda:
        devices = args.device.split(",")
        if len(devices) > 1:
            devices = tuple([int(device) for device in devices])
        else:
            devices = int(devices[0])
        args.__dict__['device'] = devices

    logger.info("Experiment Parameters")
    for arg in sorted(vars(args)):
        logger.info('{:<15}\t{}'.format(arg, getattr(args, arg)))

    model_date_dir = join(args.data_path, 'models', '{}'.format(datetime.now().strftime("%Y_%m_%d")))
    if not os.path.exists(model_date_dir):
        os.makedirs(model_date_dir)
    model_dir = join(model_date_dir, args.exp_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    return args, logger, model_dir


def setup(args, logger):
    print()
    logger.info("Loading Yamada model.")
    yamada_model = pickle_load(join(args.data_path, 'yamada', args.yamada_model))
    logger.info("Model loaded.")

    priors, _ = pickle_load(join(args.data_path, 'yamada', 'stats.pickle'))
    conditionals = pickle_load(join(args.data_path, 'necounts', 'prior_prob.pickle'))

    logger.info("Priors and conditionals loaded.")

    if args.data_type == 'conll':
        pershina = PershinaExamples(args, yamada_model)
        train_data, dev_data, test_data = pershina.get_training_examples()
    elif args.data_type == 'wiki':
        train_data, dev_data, test_data = load_data(args, yamada_model)
    else:
        logger.error("Data type {} not recognized, choose one of wiki, conll".format(args.data_type))
        sys.exit(1)

    logger.info("Training data created.")

    train_dataset = YamadaDataset(ent_conditional=conditionals,
                                  ent_prior=priors,
                                  yamada_model=yamada_model,
                                  data=train_data,
                                  args=args,
                                  cand_type=args.cand_type)
    train_loader = train_dataset.get_loader(batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            drop_last=False)

    dev_dataset = YamadaDataset(ent_conditional=conditionals,
                                ent_prior=priors,
                                yamada_model=yamada_model,
                                data=dev_data,
                                args=args,
                                cand_type=args.cand_type)
    dev_loader = dev_dataset.get_loader(batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.num_workers,
                                        drop_last=False)
    logger.info("Dataset created.")

    logger.info("There will be {} batches.".format(len(train_dataset) // args.batch_size + 1))
    validator = YamadaValidator(loader=dev_loader, args=args)
    logger.info("Validator created.")

    return train_loader, validator, yamada_model


def get_model(args, yamada_model, logger):
    if args.include_stats and args.include_string:
        model = YamadaContextStatsString(yamada_model=yamada_model, args=args)
        logger.info("Model YamadaContextStatsString created.")
    elif args.include_stats and not args.include_string:
        model = YamadaContextStats(yamada_model=yamada_model, args=args)
        logger.info("Model YamadaContextStats created.")
    elif not args.include_stats and args.include_string:
        model = YamadaContextString(yamada_model=yamada_model, args=args)
        logger.info("Model YamadaContextString created.")
    else:
        model = YamadaContext(yamada_model=yamada_model, args=args)
        logger.info("Model YamadaContext created.")

    if args.use_cuda:
        if isinstance(args.device, tuple):
            model = model.cuda(args.device[0])
            model = DataParallel(model, args.device)
        else:
            model = model.cuda(args.device)

    return model


def train(model):

    logger.info("Starting validation for untrained model.")
    correct, mentions = validator.validate(model)
    perc = correct / mentions * 100
    logger.info('Untrained, Correct : {}, Mention : {}, Percentage : {}'.format(correct, mentions, perc))

    trainer = Trainer(loader=train_loader,
                      args=args,
                      validator=validator,
                      model=model,
                      model_dir=model_dir,
                      model_type='yamada')
    logger.info("Starting Training")
    trainer.train()
    logger.info("Finished Training")


if __name__ == '__main__':
    args, logger, model_dir = parse_args()
    train_loader, validator, yamada_model = setup(args, logger)
    model = get_model(args, yamada_model, logger)
    train(model)
