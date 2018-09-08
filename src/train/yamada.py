# Training file for original yamada model
import os
from os.path import join
from datetime import datetime

import numpy as np

import torch
from torch.nn import DataParallel

import configargparse

from src.utils import str2bool
from src.data_utils import pickle_load
from src.conll.pershina import PershinaExamples
from src.dataloaders.yamada_pershina import YamadaPershina
from src.evaluation.yamada import YamadaValidator
from src.models.yamada.yamada_context import YamadaContext
from src.models.yamada.yamada_context_stats import YamadaContextStats
from src.models.yamada.yamada_context_stats_string import YamadaContextStatsString
from src.models.yamada.yamada_context_string import YamadaContextString
from src.logger import get_logger
from src.train.trainer import Trainer

np.warnings.filterwarnings('ignore')

parser = configargparse.ArgumentParser(description='Training Wikinet 2',
                                       formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
# General
general = parser.add_argument_group('General Settings.')
parser.add_argument('--my-config', required=True, is_config_file=True, help='config file path')
parser.add_argument('--seed', type=int, default=-1, help="Initialization seed")
parser.add_argument('--exp_name', type=str, default="debug", help="Experiment name")
parser.add_argument("--debug", type=str2bool, default=True, help="whether to debug")

# Data
data = parser.add_argument_group('Data Settings.')
data.add_argument('--data_path', type=str, help='location of data dir')
data.add_argument('--yamada_model', type=str, help='name of yamada model')


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

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
args.__dict__['use_cuda'] = use_cuda
logger = get_logger(args)

if args.wd > 0:
    assert not args.sparse

if use_cuda:
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

priors, conditionals = pickle_load(join(args.data_path, 'yamada', 'stats.pickle'))
logger.info("Priors and conditionals loaded.")

pershina = PershinaExamples(args, yamada_model)
train_data, dev_data, test_data = pershina.get_training_examples()
logger.info("Training data created.")

train_dataset = YamadaPershina(ent_conditional=conditionals,
                               ent_prior=priors,
                               yamada_model=yamada_model,
                               data=train_data,
                               args=args)
train_loader = train_dataset.get_loader(batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.num_workers,
                                        drop_last=False)

dev_dataset = YamadaPershina(ent_conditional=conditionals,
                             ent_prior=priors,
                             yamada_model=yamada_model,
                             data=dev_data,
                             args=args)
dev_loader = dev_dataset.get_loader(batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    drop_last=False)

full_dataset = YamadaPershina(ent_conditional=conditionals,
                              ent_prior=priors,
                              yamada_model=yamada_model,
                              data=dev_data,
                              args=args,
                              cand_rand=True)
full_loader = full_dataset.get_loader(batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    drop_last=False)

test_dataset = YamadaPershina(ent_conditional=conditionals,
                              ent_prior=priors,
                              yamada_model=yamada_model,
                              data=test_data,
                              args=args)
test_loader = test_dataset.get_loader(batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      drop_last=False)
logger.info("Dataset created.")
logger.info("There will be {} batches.".format(len(train_dataset) // args.batch_size + 1))
validator = YamadaValidator(loader=dev_loader, args=args)
full_validator = YamadaValidator(loader=full_loader, args=args)
logger.info("Validator created.")

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

if use_cuda:
    if isinstance(args.device, tuple):
        model = model.cuda(args.device[0])
        model = DataParallel(model, args.device)
    else:
        model = model.cuda(args.device)

logger.info("Starting validation for untrained model.")
correct, mentions = validator.validate(model)
perc = correct / mentions * 100
logger.info('Untrained, Correct : {}, Mention : {}, Percentage : {}'.format(correct, mentions, perc))

trainer = Trainer(loader=train_loader,
                  args=args,
                  validator=validator,
                  model=model,
                  model_dir=model_dir)
logger.info("Starting Training")
best_model = trainer.train()
logger.info("Finished Training")

logger.info("Validating on the full without pershina candidates.")
percs = []
for _ in range(10):
    correct, mentions = full_validator.validate(best_model)
    perc = correct / mentions * 100
    percs.append(perc)
    logger.info('Untrained, Correct : {}, Mention : {}, Percentage : {}'.format(correct, mentions, perc))
percs = np.array(percs)
avg = np.mean(percs)
std = np.std(percs)
logger.info("Average : {}, Std : {}".format(avg, std))
logger.info("Validation on test set.")
test_validator = YamadaValidator(loader=test_loader, args=args)
correct, mentions = test_validator.validate(model=best_model)
perc = correct / mentions * 100
logger.info('Correct : {}, Mention : {}, Percentage : {}'.format(correct, mentions, perc))