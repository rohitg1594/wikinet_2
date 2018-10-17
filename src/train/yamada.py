# Training file for original yamada model
import os
from os.path import join
from datetime import datetime
from collections import defaultdict
import configargparse

import numpy as np

from src.utils.utils import str2bool, send_to_cuda
from src.utils.data import pickle_load, load_data
from src.dataloaders.yamada import YamadaDataset
from src.eval.yamada import YamadaValidator
from src.models.models import Models
from src.logger import get_logger
from src.train.trainer import Trainer

np.warnings.filterwarnings('ignore')
DATA_TYPES = ['wiki', 'conll', 'ace2004', 'msnbc']


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
    data.add_argument('--data_type', type=str, choices=['conll', 'wiki', 'proto'], help='whether to train with conll or wiki')
    data.add_argument('--num_shards', type=int, help='number of shards of training file')
    data.add_argument('--train_size', type=int, help='number of training abstracts')

    # Max Padding
    padding = parser.add_argument_group('Max Padding for batch.')
    padding.add_argument('--max_context_size', type=int, help='max number of context')
    padding.add_argument('--max_ent_size', type=int, help='max number of entities considered in abstract')
    padding.add_argument('--num_docs', type=int, help='max number of docs to use to create corpus vec')
    padding.add_argument('--ignore_init', type=str2bool, help='whether to ignore first five tokens of context')

    # Model Type
    model_selection = parser.add_argument_group('Type of model to train.')
    model_selection.add_argument('--model_name', type=str, help='name of model to train')

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
    training = parser.add_argument_group("Training parameters.")
    training.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    training.add_argument("--save_every", type=int, default=5, help="how often to checkpoint")
    training.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    training.add_argument("--batch_size", type=int, default=32, help="Batch size")
    training.add_argument("--num_workers", type=int, default=4, help="number of workers for data loader")
    training.add_argument('--lr', type=float, help='learning rate')
    training.add_argument('--wd', type=float, help='weight decay')
    training.add_argument('--optim', type=str, choices=['adagrad', 'adam', 'rmsprop'], help='optimizer')
    training.add_argument('--sparse', type=str2bool, help='sparse gradients')

    # Loss
    loss = parser.add_argument_group('Type of loss.')
    loss.add_argument('--loss_func', type=str, default='cross_entropy', choices=['cross_entropy', 'cosine'],
                      help='loss function')
    loss.add_argument('--margin', type=float, help='margin of hinge loss')

    # cuda
    parser.add_argument("--device", type=str, help="cuda device")
    parser.add_argument("--use_cuda", type=str2bool, help="use gpu or not")
    parser.add_argument("--profile", type=str2bool, help="whether to run profiler on dataloader and exit")

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

    logger.info("Experiment Parameters:")
    print()
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
    logger.info("Loading Yamada model.....")
    yamada_model = pickle_load(join(args.data_path, 'yamada', args.yamada_model))
    logger.info("Model loaded.")

    logger.info("Loading Stat features.....")
    priors, _ = pickle_load(join(args.data_path, 'yamada', 'stats.pickle'))
    conditionals = pickle_load(join(args.data_path, 'necounts', 'prior_prob.pickle'))
    logger.info("Priors and conditionals loaded.")

    logger.info("Using {} for training.....".format(args.data_type))
    data = defaultdict(dict)

    for data_type in DATA_TYPES:
        if data_type == 'wiki':
            train, dev, test = load_data('proto_370k', args)
            dev = dev[:5000]
            data['wiki'] = {}
            data['wiki']['train'] = train
            data['wiki']['dev'] = dev
            data['wiki']['test'] = test
        if data_type == 'conll':
            for split in ['train', 'dev', 'test']:
                data['conll'][split] = pickle_load(join(args.data_path, f'training_files/conll-{split}.pickle'))
        else:
            data[data_type]['dev'] = pickle_load(join(args.data_path, f'training_files/{data_type}.pickle'))

    if args.data_type == 'conll':
        train_data = data['conll']['train']
    else:
        train_data = data['wiki']['train']
    logger.info("Data loaded.")

    logger.info("Creating data loaders.....")
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
    validators = {}
    for data_type in DATA_TYPES:
        dataset = YamadaDataset(ent_conditional=conditionals,
                                ent_prior=priors,
                                yamada_model=yamada_model,
                                data=data[data_type]['dev'],
                                args=args,
                                cand_type='necounts')
        loader = dataset.get_loader(batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    drop_last=False)
        validators[data_type] = YamadaValidator(loader=loader, args=args,
                                                word_dict=yamada_model['word_dict'],
                                                ent_dict=yamada_model['ent_dict'])

    logger.info("Data loaders and validators created.There will be {} batches.".format(len(train_loader)))

    return train_loader, validators, yamada_model


def get_model(args, yamada_model, logger):
    model_type = getattr(Models, args.model_name)
    model = model_type(yamada_model=yamada_model, args=args)

    if args.use_cuda:
        model = send_to_cuda(args.device, model)
    logger.info('{} Model created.'.format(model_type.__name__))

    return model


def train(model=None,
          logger=None,
          validators=None,
          model_dir=None,
          train_loader=None,
          args=None):

    logger.info("Starting validation for untrained model.")
    for data_type in DATA_TYPES:
        correct, mentions = validators[data_type].validate(model)
        res = correct / mentions * 100
        logger.info(f'Untrained, {data_type} - {res}')

    trainer = Trainer(loader=train_loader,
                      args=args,
                      validator=validators,
                      model=model,
                      model_dir=model_dir,
                      model_type='yamada',
                      profile=args.profile)
    logger.info("Starting Training:")
    print()
    trainer.train()
    logger.info("Finished Training")


if __name__ == '__main__':
    Args, Logger, Model_dir = parse_args()
    Train_loader, Validators, Yamada_model = setup(Args, Logger)
    Model = get_model(Args, Yamada_model, Logger)
    train(model=Model,
          validators=Validators,
          model_dir=Model_dir,
          train_loader=Train_loader,
          logger=Logger,
          args=Args)
