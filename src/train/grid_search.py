# Main training file
import os
from os.path import join
from datetime import datetime

import numpy as np

import configargparse

from src.utils.utils import str2bool, normal_initialize, get_model
from src.utils.data import load_vocab, pickle_load, load_data
from src.eval.combined import CombinedValidator
from src.dataloaders.combined import CombinedDataSet
from src.tokenizer.gram_tokenizer import get_gram_tokenizer
from src.logger import get_logger
from src.train.trainer import Trainer

np.warnings.filterwarnings('ignore')

parser = configargparse.ArgumentParser(description='Training Wikinet 2', formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
# General
general = parser.add_argument_group('General Settings.')
parser.add_argument('--my-config', required=True, is_config_file=True, help='config file path')
parser.add_argument('--seed', type=int, default=-1, help="Initialization seed")
parser.add_argument('--exp_name', type=str, default="debug", help="Experiment name")
parser.add_argument("--debug", type=str2bool, default=True, help="whether to debug")

# Data
data = parser.add_argument_group('Data Settings.')
data.add_argument('--data_path', type=str, help='location of data dir')
data.add_argument('--data_type', choices=['wiki', 'conll'], type=str, help='dataset to train on.')
data.add_argument('--proto_data', type=str2bool, help='whether to use prototype data')
data.add_argument('--num_shards', type=int, help='number of shards of training file')
data.add_argument('--train_size', type=int, help='number of training abstracts')
data.add_argument('--query_size', type=int, help='number of queries during validation')
data.add_argument('--conll_split', type=str, choices=['train', 'dev', 'test'],  help='which split of connl data to evaluate on')
data.add_argument('--yamada_model', type=str, help='name of yamada model')

# Gram
gram = parser.add_argument_group('Gram (uni / bi / tri) Settings.')
gram.add_argument('--gram_type', type=str, choices=['unigram', 'bigram', 'trigram'], help='type of gram tokenizer')
gram.add_argument('--gram_lower', type=str2bool, help='whether to lowercase gram tokens')
gram.add_argument('--gram_vocab', type=str, help='name of gram vocab file')
gram.add_argument('--gram_dim', type=int, help='dimension of gram embeddings')

# Max Padding
padding = parser.add_argument_group('Max Padding for batch.')
padding.add_argument('--max_word_size', type=int, help='max number of words')
padding.add_argument('--max_context_size', type=int, help='max number of context')
padding.add_argument('--max_gram_size', type=int, help='max number of grams')
padding.add_argument('--max_ent_size', type=int, help='max number of entities considered in abstract')

# Model Type
model_selection = parser.add_argument_group('Type of model to train.')
model_selection.add_argument('--init_yamada', type=str2bool, help='whether to initialize the combined model randomly')
model_names = ['only_prior', 'only_prior_linear', 'include_word', 'include_gram', 'mention_prior', 'weigh_concat']
model_selection.add_argument('--model_name', type=str, choices=model_names, help='type of model to train')
model_selection.add_argument('--init_mention', type=str, help='how to initialize mention and ent mention embs')
model_selection.add_argument('--init_mention_model', type=str, help='ckpt file to initialize mention and ent mention embs')

# Model params
model_params = parser.add_argument_group("Parameters for chosen model.")
model_params.add_argument('--mention_word_dim', type=int, help='dimension of mention word embeddings')
model_params.add_argument('--measure', type=str, default='ip', choices=['ip', 'l2'], help='faiss index')
model_params.add_argument('--dp', type=float, help='drop out')

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
train = parser.add_argument_group("Training parameters.")
train.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
train.add_argument("--bold_driver", type=str2bool, default=False, help="whether to use bold driver heuristic to adjust lr")
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
loss.add_argument('--loss_func', type=str, default='cross_entropy', choices=['cross_entropy', 'cosine'], help='loss function')
loss.add_argument('--margin', type=float, help='margin of hinge loss')

# Things to Train
train_selection = parser.add_argument_group('Parameters to train')
train_selection.add_argument('--train_word', type=str2bool, help='whether to train word embeddings')
train_selection.add_argument('--train_ent', type=str2bool, help='whether to train entity embeddings')
train_selection.add_argument('--train_gram', type=str2bool, help='whether to train gram embeddings')
train_selection.add_argument('--train_mention', type=str2bool, help='whether to train mention word embeddings')
train_selection.add_argument('--train_linear', type=str2bool, help='whether to train linear layer')

# cuda
parser.add_argument("--device", type=str, help="cuda device")
parser.add_argument("--use_cuda", type=str2bool, help="whether to use cuda")


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

print()
logger.info("Loading Yamada model.")
yamada_model = pickle_load(join(args.data_path, 'yamada', args.yamada_model))
logger.info("Model loaded.")

# Gram Embeddings
gram_tokenizer = get_gram_tokenizer(gram_type=args.gram_type, lower_case=args.gram_lower)
logger.info("Using gram tokenizer {}".format(gram_tokenizer.__name__))
gram_dict = load_vocab(join(args.data_path, 'gram_vocabs', args.gram_vocab), plus_one=True)
logger.info("Gram dictionary loaded of length: {}".format(len(gram_dict)))
gram_embs = normal_initialize(len(gram_dict) + 1, args.gram_dim)
logger.info("Gram embeddings created of shape: {}".format(gram_embs.shape))

# Word and Entity Embeddings
if not args.init_yamada:
    logger.info("Initializing word and entity embeddings randomly...")
    word_embs = normal_initialize(yamada_model['word_emb'].shape[0], yamada_model['word_emb'].shape[1])
    ent_embs = normal_initialize(yamada_model['ent_emb'].shape[0], yamada_model['ent_emb'].shape[1])
else:
    logger.info("Using pre-trained word and entity embeddings from Yamada.")
    ent_embs = yamada_model['ent_emb']
    word_embs = yamada_model['word_emb']

# Training Data
train_data, dev_data, test_data = load_data(args, yamada_model)
logger.info("Training data loaded.")
logger.info("Train : {}, Dev : {}, Test :{}".format(len(train_data), len(dev_data), len(test_data)))

# Validation
validator = CombinedValidator(gram_dict=gram_dict,
                              gram_tokenizer=gram_tokenizer,
                              yamada_model=yamada_model,
                              data=dev_data,
                              args=args)
logger.info("Validator created.")

# Dataset
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
# train_loader = gen_wrapper(iter(train_loader))
logger.info("Dataset created.")
logger.info("There will be {} batches.".format(len(train_dataset) // args.batch_size + 1))

for norm_final in [True, False]:
    for measure in ['ip', 'l2']:
            args.__dict__['norm_final'] = norm_final
            args.__dict__['measure'] = measure
            logger.info("GRID SEARCH PARAMS : norm_final - {}, measure - {}".format(norm_final, measure))

            # Model
            model = get_model(args,
                              yamada_model=yamada_model,
                              ent_embs=ent_embs,
                              word_embs=word_embs,
                              gram_embs=gram_embs,
                              init=args.init_mention)

            logger.info("Starting validation for untrained model.")
            top1_wiki, top10_wiki, top100_wiki, mrr_wiki, top1_conll, top10_conll, top100_conll, mrr_conll = validator.validate(model=model)
            logger.info('Dev Validation')
            logger.info("Wikipedia, Untrained Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(top1_wiki, top10_wiki, top100_wiki, mrr_wiki))
            logger.info("Conll, Untrained Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(top1_conll, top10_conll, top100_conll, mrr_conll))

            trainer = Trainer(loader=train_loader,
                              args=args,
                              validator=validator,
                              model=model,
                              model_dir=model_dir,
                              model_type='combined')
            logger.info("Starting Training")
            trainer.train()
            logger.info("Finished Training")
