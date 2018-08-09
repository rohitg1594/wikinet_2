# Main training file
import os
from os.path import join

import numpy as np

import torch

import configargparse

from src.utils import str2bool, normal_initialize
from src.data_utils import load_yamada, load_vocab, pickle_load
from src.evaluation.validation import Validator
from src.dataloaders.combined import CombinedDataSet
from src.tokenization.gram_tokenizer import get_gram_tokenizer
from src.models.context_gram import ContextGramModel
from src.models.context_gram_word import ContextGramWordModel
from src.logger import get_logger
from src.trainer import Trainer

np.warnings.filterwarnings('ignore')

# main
parser = configargparse.ArgumentParser(description='Training Wikinet 2',
                                       formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--my-config', required=True, is_config_file=True, help='config file path')
parser.add_argument('--seed', type=int, default=-1, help="Initialization seed")
parser.add_argument('--exp_name', type=str, default="debug", help="Experiment name")
# debug
parser.add_argument("--debug", type=str2bool, default=True, help="whether to debug")
# data
parser.add_argument('--data_path', type=str, help='location of data dir')
parser.add_argument('--yamada_model', type=str, help='name of yamada model')
parser.add_argument('--num_shards', type=int, help='number of shards of training file')
parser.add_argument('--gram_type', type=str, choices=['unigram', 'bigram', 'trigram'], help='type of gram tokenization')
parser.add_argument('--gram_vocab', type=str, help='name of gram vocab file')
parser.add_argument('--train_size', type=int, help='number of training abstracts')
# validation
parser.add_argument('--query_size', type=int, help='number of queries during validation')
parser.add_argument('--conll_split', type=str, choices=['train', 'dev', 'test'],  help='which split of connl data to evaluate on')
# model max padding sizes
parser.add_argument('--max_word_size', type=int, help='max number of words')
parser.add_argument('--max_context_size', type=int, help='max number of context')
parser.add_argument('--max_gram_size', type=int, help='max number of grams')
parser.add_argument('--max_ent_size', type=int, help='max number of entities considered in abstract')
# model type
parser.add_argument('--include_word', type=str2bool, help='whether to include word information')
parser.add_argument('--include_gram', type=str2bool, help='whether to include word information')
parser.add_argument('--include_context', type=str2bool, help='whether to include word information')
parser.add_argument('--norm_gram', type=str2bool, help='whether to normalize gram embs')
parser.add_argument('--norm_word', type=str2bool, help='whether to normalize word embs')
parser.add_argument('--norm_context', type=str2bool, help='whether to normalize context embs')
parser.add_argument('--norm_final', type=str2bool, help='whether to normalize final embs')
# model hyperparameters
parser.add_argument('--cand_gen_rand', type=str2bool, help='whether to generate random candidates')
parser.add_argument("--num_candidates", type=int, default=32, help="Number of candidates")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loader")
parser.add_argument('--gram_dim', type=int, help='dimension of gram embeddings')
parser.add_argument('--loss_func', type=str, default='cross_entropy', choices=['cross_entropy', 'cosine'], help='loss function')
parser.add_argument('--margin', type=float, help='margin of hinge loss')
parser.add_argument('--measure', type=str, default='ip', choices=['ip', 'l2'], help='faiss index')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--wd', type=float, help='weight decay')
parser.add_argument('--optim', type=str, choices=['adagrad', 'adam'], help='optimizer')
parser.add_argument('--sparse', type=str2bool, help='sparse gradients')
# paramters to train
parser.add_argument('--train_word', type=str2bool, help='whether to train word embeddings')
parser.add_argument('--train_ent', type=str2bool, help='whether to train entity embeddings')
parser.add_argument('--train_gram', type=str2bool, help='whether to train gram embeddings')
parser.add_argument('--train_linear', type=str2bool, help='whether to train linear layer')
# cuda
parser.add_argument("--device", type=int, help="cuda device")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

if args.wd > 0:
    assert not args.sparse
logger = get_logger(args)

logger.info("Experiment Parameters")
for arg in vars(args):
    logger.info('{:<15}\t{}'.format(arg, getattr(args, arg)))
logger.info('{:<15}\t{}'.format('cuda available', use_cuda))

model_dir = join(args.data_path, 'models', args.exp_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logger.info("Loading Yamada model.")
yamada_model = load_yamada(join(args.data_path, 'yamada', args.yamada_model))
logger.info("Model loaded.")

# Gram
gram_tokenizer = get_gram_tokenizer(gram_type=args.gram_type)
gram_vocab = load_vocab(join(args.data_path, 'gram_vocabs', args.gram_vocab), plus_one=True)
gram_embs = normal_initialize(len(gram_vocab) + 1, args.gram_dim)

# Training Data
logger.info("Loading Training data.")
data = []
for i in range(args.num_shards):
    data.extend(pickle_load(join(args.data_path, 'training-yamada-simple', 'data_{}.pickle'.format(i))))

train_data = []
dev_data = []
test_data = []
for d in data:
    if len(train_data) == args.train_size:
        break
    r = np.random.random()
    if r < 0.8:
        train_data.append(d)

    elif 0.8 < r < 0.9:
        dev_data.append(d)

    else:
        test_data.append(d)

logger.info("Training data loaded.")
logger.info("Train : {}, Dev : {}, Test :{}".format(len(train_data), len(dev_data), len(test_data)))

# Validation
dev_validator = Validator(gram_dict=gram_vocab,
                          gram_tokenizer=gram_tokenizer,
                          yamada_model=yamada_model,
                          data=dev_data,
                          args=args)
logger.info("Validators created.")

# Dataset
train_dataset = CombinedDataSet(gram_tokenizer=gram_tokenizer,
                                gram_vocab=gram_vocab,
                                word_vocab=yamada_model['word_dict'],
                                ent2id=yamada_model['ent_dict'],
                                data=train_data,
                                args=args)
train_loader = train_dataset.get_loader(batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.num_workers,
                                        drop_last=True)
logger.info("Dataset created.")

# Model
if args.include_word:
    model = ContextGramWordModel(yamada_model=yamada_model, gram_embs=gram_embs, args=args)
else:
    model = ContextGramModel(yamada_model=yamada_model, gram_embs=gram_embs, args=args)
if use_cuda:
    model = model.cuda(args.device)
logger.info('Model created.')

logger.info("Starting validation for untrained model.")
top1_wiki, top10_wiki, top100_wiki, mrr_wiki, top1_conll, top10_conll, top100_conll, mrr_conll = train_validator.validate(model=model)
logger.info('Train Validation')
logger.info("Wikipedia, Untrained Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(top1_wiki, top10_wiki, top100_wiki, mrr_wiki))
logger.info("Conll, Untrained Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(top1_conll, top10_conll, top100_conll, mrr_conll))

top1_wiki, top10_wiki, top100_wiki, mrr_wiki, top1_conll, top10_conll, top100_conll, mrr_conll = dev_validator.validate(model=model)
logger.info('Dev Validation')
logger.info("Wikipedia, Untrained Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(top1_wiki, top10_wiki, top100_wiki, mrr_wiki))
logger.info("Conll, Untrained Top 1 - {:.4f}, Top 10 - {:.4f}, Top 100 - {:.4f}, MRR - {:.4f}".format(top1_conll, top10_conll, top100_conll, mrr_conll))

trainer = Trainer(loader=train_loader,
                  args=args,
                  dev_validator=dev_validator,
                  model=model,
                  model_dir=model_dir,
                  use_cuda=use_cuda)
logger.info("Starting Training")
trainer.train()
logger.info("Finished Training")
