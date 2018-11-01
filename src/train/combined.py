# Main training file
import os
from os.path import join
from datetime import datetime

import numpy as np

from torch.nn import DataParallel

import configargparse

from src.utils.utils import str2bool, normal_initialize, get_model, send_to_cuda
from src.utils.data import load_vocab, pickle_load, load_data
from src.eval.combined import CombinedValidator
from src.dataloaders.combined import CombinedDataSet
from src.tokenizer.gram_tokenizer import get_gram_tokenizer
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
    general.add_argument('--debug', type=str2bool, default=True, help="whether to debug")
    general.add_argument('--error', type=str2bool, default=True, help="whether to print out errors after every epoch.")

    # Data
    data = parser.add_argument_group('Data Settings.')
    data.add_argument('--data_path', type=str, help='location of data dir')
    data.add_argument('--data_type', type=str, help='dataset to train on.')
    data.add_argument('--num_shards', type=int, help='number of shards of training file')
    data.add_argument('--train_size', type=int, help='number of training abstracts')
    data.add_argument('--query_size', type=int, help='number of queries during validation')
    data.add_argument('--conll_split', type=str, choices=['train', 'dev', 'test'],
                      help='which split of connl data to evaluate on')
    data.add_argument('--yamada_model', type=str, help='name of yamada model')

    # Gram
    gram = parser.add_argument_group('Gram (uni / bi / tri) Settings.')
    gram.add_argument('--gram_type', type=str, choices=['unigram', 'bigram', 'trigram'], help='type of gram tokenizer')
    gram.add_argument('--gram_lower', type=str2bool, help='whether to lowercase gram tokens')
    gram.add_argument('--gram_dim', type=int, help='dimension of gram embeddings')

    # Max Padding
    padding = parser.add_argument_group('Max Padding for batch.')
    padding.add_argument('--max_word_size', type=int, help='max number of words')
    padding.add_argument('--max_context_size', type=int, help='max number of context')
    padding.add_argument('--max_gram_size', type=int, help='max number of grams')
    padding.add_argument('--max_ent_size', type=int, help='max number of entities considered in abstract')
    padding.add_argument('--ignore_init', type=str2bool, help='whether to ignore first five tokens of context')

    # Model Type
    model_selection = parser.add_argument_group('Type of model to train.')
    model_selection.add_argument('--init_yamada', type=str2bool,
                                 help='whether to initialize the combined model randomly')
    model_selection.add_argument('--model_name', type=str, help='type of model to train')
    model_selection.add_argument('--init_mention', type=str, help='how to initialize mention and ent mention embs')
    model_selection.add_argument('--init_mention_model', type=str,
                                 help='ckpt file to initialize mention and ent mention embs')

    # Model params
    model_params = parser.add_argument_group("Parameters for chosen model.")
    model_params.add_argument('--mention_word_dim', type=int, help='dimension of mention word embeddings')
    model_params.add_argument('--context_word_dim', type=int, help='dimension of mention word embeddings')
    model_params.add_argument('--ent_mention_dim', type=int, help='dimension of mention entity embeddings')
    model_params.add_argument('--measure', type=str, default='ip', choices=['ip', 'l2'], help='faiss index')
    model_params.add_argument('--dp', type=float, help='drop out')
    model_params.add_argument('--init_stdv', type=float,
                              help='standard deviation to initialize embeddings in small context')
    model_params.add_argument('--combined_linear', type=str2bool,
                              help='whether to have a combining linear layer in small context model')

    # Normalization
    normal = parser.add_argument_group('Which embeddings to normalize?')
    normal.add_argument('--norm_gram', type=str2bool, help='whether to normalize gram embs')
    normal.add_argument('--norm_mention', type=str2bool, help='whether to normalize mention word embs')
    normal.add_argument('--norm_word', type=str2bool, help='whether to normalize word embs')
    normal.add_argument('--norm_context', type=str2bool, help='whether to normalize context embs')
    normal.add_argument('--norm_final', type=str2bool, help='whether to normalize final embs')

    # Candidate Generation
    candidate = parser.add_argument_group('Candidate generation.')
    candidate.add_argument('--cand_gen_rand', type=str2bool, help='whether to generate random candidates')
    candidate.add_argument("--num_candidates", type=int, default=32, help="Number of candidates")

    # Training
    train_params = parser.add_argument_group("Training parameters.")
    train_params.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    train_params.add_argument("--bold_driver", type=str2bool, default=False,
                              help="whether to use bold driver heuristic to adjust lr")
    train_params.add_argument("--save_every", type=int, default=5, help="how often to checkpoint")
    train_params.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    train_params.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_params.add_argument("--num_workers", type=int, default=4, help="number of workers for data loader")
    train_params.add_argument('--lr', type=float, help='learning rate')
    train_params.add_argument('--wd', type=float, help='weight decay')
    train_params.add_argument('--optim', type=str, choices=['adagrad', 'adam', 'rmsprop'], help='optimizer')
    train_params.add_argument('--sparse', type=str2bool, help='sparse gradients')

    # Loss
    loss = parser.add_argument_group('Type of loss.')
    loss.add_argument('--loss_func', type=str, default='cross_entropy', choices=['cross_entropy', 'cosine'],
                      help='loss function')
    loss.add_argument('--margin', type=float, help='margin of hinge loss')

    # Things to Train
    train_selection = parser.add_argument_group('Parameters to train')
    train_selection.add_argument('--train_word', type=str2bool, help='whether to train word embeddings')
    train_selection.add_argument('--train_ent', type=str2bool, help='whether to train entity embeddings')
    train_selection.add_argument('--train_gram', type=str2bool, help='whether to train gram embeddings')
    train_selection.add_argument('--train_mention', type=str2bool, help='whether to train mention word embeddings')
    train_selection.add_argument('--train_linear', type=str2bool, help='whether to train linear layer')

    # cuda and profiler
    parser.add_argument("--device", type=str, help="cuda device")
    parser.add_argument("--use_cuda", type=str2bool, help="use gpu or not")
    parser.add_argument("--profile", type=str2bool, help="whether to run profiler on dataloader and exit")

    args = parser.parse_args()
    logger = get_logger(args)

    # Setup
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


def setup(args=None, logger=None):
    # Yamada model
    print()
    logger.info("Loading Yamada model.")
    yamada_model = pickle_load(join(args.data_path, 'yamada', args.yamada_model))
    logger.info("Model loaded.")

    # Gram Embeddings
    gram_tokenizer = get_gram_tokenizer(gram_type=args.gram_type, lower_case=args.gram_lower)
    logger.info(f"Using gram tokenizer {gram_tokenizer.__name__}")
    gram_dict = load_vocab(join(args.data_path, 'gram_vocabs', args.gram_type + '.tsv'), plus_one=True)
    logger.info(f"Gram dictionary loaded of length: {len(gram_dict)}")
    gram_embs = normal_initialize(len(gram_dict) + 1, args.gram_dim)
    logger.info(f"Gram embeddings created of shape: {gram_embs.shape}")

    # Word and Entity Embeddings
    if not args.init_yamada:
        logger.info("Initializing word and entity embeddings randomly.....")
        word_embs = normal_initialize(yamada_model['word_emb'].shape[0], yamada_model['word_emb'].shape[1])
        ent_embs = normal_initialize(yamada_model['ent_emb'].shape[0], yamada_model['ent_emb'].shape[1])
        logger.info("Embeddings initialized.")
    else:
        logger.info("Using pre-trained word and entity embeddings from Yamada.")
        ent_embs = yamada_model['ent_emb']
        word_embs = yamada_model['word_emb']

    # Training Data
    logger.info("Loading training data.....")
    res = load_data(args.data_type, args)
    train_data, dev_data, test_data = res['train'], res['dev'], res['test']
    logger.info("Training data loaded.")
    logger.info(f"Train : {len(train_data[1])}, Dev : {len(dev_data[1])}, Test :{len(test_data)}")

    # Validation
    logger.info("Creating validator.....")
    validator = CombinedValidator(gram_dict=gram_dict,
                                  gram_tokenizer=gram_tokenizer,
                                  yamada_model=yamada_model,
                                  data=dev_data,
                                  args=args)
    logger.info("Validator created.")

    # Dataset
    logger.info("Creating Dataset.....")
    train_dataset = CombinedDataSet(gram_tokenizer=gram_tokenizer,
                                    gram_dict=gram_dict,
                                    word_dict=yamada_model['word_dict'],
                                    ent_dict=yamada_model['ent_dict'],
                                    data=train_data,
                                    args=args)
    train_loader = train_dataset.get_loader(batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            drop_last=True)
    logger.info("Dataset created.")
    logger.info(f"There will be {len(train_dataset) // args.batch_size + 1} batches.")

    return train_loader, validator, yamada_model, ent_embs, word_embs, gram_embs


def train(args=None,
          yamada_model=None,
          ent_embs=None,
          word_embs=None,
          gram_embs=None,
          validator=None,
          logger=None,
          train_loader=None,
          model_dir=None):
    # Model
    model = get_model(args,
                      yamada_model=yamada_model,
                      ent_embs=ent_embs,
                      word_embs=word_embs,
                      gram_embs=gram_embs)
    if args.use_cuda:
        model = send_to_cuda(args.device, model)

    logger.info("Validating untrained model.....")
    results = validator.validate(model=model, error=args.error)
    for data_type in ['wiki', 'conll', 'msnbc', 'ace2004']:
        res_str = ""
        for k, v in results[data_type].items():
            res_str += k.upper() + ': {:.3},'.format(v)
        logger.info(f"{data_type}: Untrained," + res_str[:-1])
    logger.info("Done validating.")

    # Train
    trainer = Trainer(loader=train_loader,
                      args=args,
                      validator=validator,
                      model=model,
                      model_dir=model_dir,
                      model_type='combined',
                      profile=args.profile)
    logger.info("Training.....")
    trainer.train()
    logger.info("Finished Training")


if __name__ == '__main__':
    Args, Logger, Model_dir = parse_args()
    Train_loader, Validator, Yamada_model, Ent_embs, Word_embs, Gram_embs = setup(args=Args, logger=Logger)
    train(args=Args,
          yamada_model=Yamada_model,
          ent_embs=Ent_embs,
          word_embs=Word_embs,
          gram_embs=Gram_embs,
          validator=Validator,
          logger=Logger,
          train_loader=Train_loader,
          model_dir=Model_dir)
