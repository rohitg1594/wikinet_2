# Main training file
import os
from os.path import join

import numpy as np
np.warnings.filterwarnings('ignore')

import torch
from torch.autograd import Variable

import configargparse

from src.utils import str2bool, normal_initialize
from src.data_utils import load_yamada, load_vocab, pickle_load, save_checkpoint
from src.evaluation.validation import Validator
from src.dataloaders.combined import CombinedDataSet
from src.tokenization.gram_tokenizer import get_gram_tokenizer
from src.models.context_gram import ContextGramModel
from src.models.context_gram_word import ContextGramWordModel
from src.logger import get_logger

# main
parser = configargparse.ArgumentParser(description='Training Wikinet 2',
                                       formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--my-config', required=True, is_config_file=True, help='config file path')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
# debug
parser.add_argument("--debug", type=str2bool, default=True, help="whether to debug")
# data
parser.add_argument('--data_path', type=str, help='location of data dir')
parser.add_argument('--yamada_model', type=str, help='name of yamada model')
parser.add_argument('--num_shards', type=int, help='number of shards of training file')
parser.add_argument('--gram_type', type=str, choices=['unigram', 'bigram', 'trigram'], help='type of gram tokenization')
parser.add_argument('--gram_vocab', type=str, help='name of gram vocab file')
# train_size
parser.add_argument('--train_size', type=int, help='number of training abstracts')
# validation
parser.add_argument('--query_size', type=int, help='number of queries during validation')
# model max padding sizes
parser.add_argument('--max_word_size', type=int, help='max number of words')
parser.add_argument('--max_context_size', type=int, help='max number of context')
parser.add_argument('--max_gram_size', type=int, help='max number of grams')
parser.add_argument('--max_ent_size', type=int, help='max number of entities considered in abstract')
# model type
parser.add_argument('--include_word', type=str2bool, help='whether to include word information')
parser.add_argument('--norm_gram', type=str2bool, help='whether to normalize gram embs')
parser.add_argument('--norm_word', type=str2bool, help='whether to normalize word embs')
parser.add_argument('--norm_context', type=str2bool, help='whether to normalize context embs')
parser.add_argument('--norm_final', type=str2bool, help='whether to normalize final embs')
# model hyperparameters
parser.add_argument('--cand_gen_rand', type=str2bool, help='whether to generate random candidates')
parser.add_argument("--num_candidates", type=int, default=32, help="Number of candidates")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loader")
parser.add_argument('--gram_dim', type=int, help='dimension of gram embeddings')
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

for arg in vars(args):
    print('{:<15}\t{}'.format(arg, getattr(args, arg)))
print('{:<15}\t{}'.format('cuda available', use_cuda))

model_dir = join(args.data_path, 'models', args.exp_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logger = get_logger(args)
logger.info("Loading Yamada model.")
yamada_model = load_yamada(join(args.data_path, 'yamada', args.yamada_model))
logger.info("Model loaded.")

# Gram
gram_tokenizer = get_gram_tokenizer(gram_type=args.gram_type)
gram_vocab = load_vocab(join(args.data_path, 'gram_vocabs', args.gram_vocab), plus_one=True)

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

# Validation
validator = Validator(gram_dict=gram_vocab,
                      gram_tokenizer=gram_tokenizer,
                      yamada_model=yamada_model,
                      data=dev_data,
                      args=args)
logger.info("Validator Created")

gram_embs = normal_initialize(len(gram_vocab) + 1, args.gram_dim)

if args.include_word:
    model = ContextGramWordModel(yamada_model=yamada_model, gram_embs=gram_embs, args=args)
else:
    model = ContextGramModel(yamada_model=yamada_model, gram_embs=gram_embs, args=args)
logger.info('Model created.')

if use_cuda:
    model = model.cuda(args.device)

if args.optim == 'adagrad':
    optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr,
                                    weight_decay=args.wd)
elif args.optim == 'adam':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr,
                                 weight_decay=args.wd)
losses = []
best_model = model
best_mrr = 0

logger.info("Starting validation for untrained model.")
top1, top10, top100, mrr = validator.validate(model=best_model,
                                              error=args.debug,
                                              gram=True,
                                              word=args.include_word,
                                              context=True,
                                              norm_gram=args.norm_gram,
                                              norm_word=args.norm_word,
                                              norm_context=args.norm_context,
                                              norm_final=args.norm_final,
                                              verbose=args.debug,
                                              measure=args.measure)
logger.info("Untrained Performance : Top 1 - {}, Top 10 - {}, Top 100 - {}, MRR - {}".format(top1, top10, top100, mrr))

logger.info("Starting Training")
for epoch in range(args.num_epochs):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        data = list(data)
        ymask = data[0]
        data = data[1:]
        for i in range(len(data)):
            data[i] = Variable(data[i])
    
        ymask = ymask.view(args.batch_size * args.max_ent_size)
        zeros_2d = Variable(torch.zeros(args.batch_size * args.max_ent_size, args.num_candidates - 1))

        if use_cuda:
            for i in range(len(data)):
                data[i] = data[i].cuda(args.device)
            ymask = ymask.cuda(args.device)
            zeros_2d = zeros_2d.cuda(args.device)

        optimizer.zero_grad()
        scores = model(tuple(data))

        scores_pos = scores[:, 0]
        scores_neg = scores[:, 1:]

        distance_pos = 1 - scores_pos
        distance_neg = torch.max(zeros_2d, scores_neg - args.margin)

        ymask_2d = ymask.repeat(args.num_candidates - 1).view(args.num_candidates - 1, -1).transpose(0, 1)
        distance_pos_masked = distance_pos * ymask
        distance_neg_masked = distance_neg * ymask_2d

        loss_pos = distance_pos_masked.sum() / ymask.sum()
        loss_neg = distance_neg_masked.sum() / ymask_2d.sum()
        loss = loss_pos + loss_neg
        # loss = -loss

        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        losses.append(loss.data[0])

    logger.info('Epoch - {}, Loss - {:.4}'.format(epoch, loss.data[0]))
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, True, filename=join(model_dir, '{}.ckpt'.format(epoch)))
    top1, top10, top100, mrr = validator.validate(model=best_model,
                                                  error=args.debug,
                                                  gram=True,
                                                  word=args.include_word,
                                                  context=True,
                                                  norm_gram=args.norm_gram,
                                                  norm_word=args.norm_word,
                                                  norm_context=args.norm_context,
                                                  norm_final=args.norm_final,
                                                  verbose=args.debug,
                                                  measure=args.measure)

    logger.info(
        "Epoch : Top 1 - {}, Top 10 - {}, Top 100 - {}, MRR - {}".format(top1, top10, top100, mrr))

    if mrr > best_mrr:
        best_model = model
        best_mrr = mrr

print('Finished Training')

save_checkpoint({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, True, filename=join(model_dir, 'final_model.ckpt'))
