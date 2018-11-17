# Main training file
from datetime import datetime
import configargparse
from copy import deepcopy

import torch.optim as optim

from src.utils.utils import *
from src.utils.dictionary import Dictionary  # needed because of autoencoder
from src.eval.autoencoder import AutoencoderValidator
from src.models.combined.string_autoencoder import StringAutoEncoder
from src.logger import get_logger

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
    data.add_argument('--yamada_model', type=str, help='name of yamada model')
    data.add_argument('--data_type', type=str, help='dataset to train on.')
    data.add_argument('--train_size', type=int, help='number of training abstracts')
    data.add_argument('--eval_sample', type=int, help='number of strs to evaluate')

    # Max Padding
    padding = parser.add_argument_group('Max Padding for batch.')
    padding.add_argument('--max_char_size', type=int, help='max number of words')

    # Model params
    model_params = parser.add_argument_group("Parameters for chosen model.")
    model_params.add_argument('--char_dim', type=int, help='dimension of char embeddings')
    model_params.add_argument('--hidden_size', type=int, help='latent code size')
    model_params.add_argument('--dp', type=float, help='drop out')
    model_params.add_argument('--norm', type=str2bool, help='whether to normalize latent code')
    model_params.add_argument('--activate', type=str, help='activation function after dropout')
    model_params.add_argument('--measure', type=str, default='ip', choices=['ip', 'l2'], help='faiss index')

    # Training
    train_params = parser.add_argument_group("Training parameters.")
    train_params.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    train_params.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_params.add_argument('--lr', type=float, help='learning rate')
    train_params.add_argument('--wd', type=float, help='weight decay')

    # cuda and profiler
    parser.add_argument("--device", type=str, help="cuda device")
    parser.add_argument("--use_cuda", type=str2bool, help="use gpu or not")
    parser.add_argument("--profile", type=str2bool, help="whether to run profiler on dataloader and exit")

    args = parser.parse_args()
    logger = get_logger(args)

    # Setup
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


def setup(args=None, logger=None, model_dir=None):

    print()

    # Training Data
    logger.info("Loading training data.....")
    data = pickle_load(join(args.data_path, 'autoencoder/data.pickle'))
    dev_arr = data['dev']
    train_arr = data['train']
    dev_strs = data['dev_strs']
    char_dict = data['char_dict']
    mention_arr = data['mention_arr']
    ent_arr = data['ent_arr']
    gold = data['gold']
    logger.info("Training data loaded.")

    # Validator
    validator = AutoencoderValidator(dev_strs=dev_strs,
                                     num_clusters=10,
                                     num_members=20,
                                     char_dict=char_dict,
                                     rank_sample=1000,
                                     verbose=True,
                                     ent_arr=ent_arr,
                                     mention_arr=mention_arr,
                                     args=args,
                                     dev_arr=dev_arr,
                                     gold=gold,
                                     model_dir=model_dir)

    return validator, char_dict, train_arr


def train_epoch(model, optimizer, data, args):
    model.train()
    full_loss = 0

    for batch_idx, batch in enumerate(chunks(data, args.batch_size)):
        batch = torch.from_numpy(batch)

        if args.use_cuda:
            batch = batch.cuda(args.device)
        input, hidden, output = model(batch)
        loss = mse(input, output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        full_loss += loss.item()

    return full_loss / (batch_idx + 1)


def train(args=None,
          validator=None,
          logger=None,
          char_dict=None,
          train_arr=None,
          model_dir=None):

    logger.info("Inititalizing char embs.....")
    char_embs = normal_initialize(len(char_dict), args.char_dim)
    logger.info(f"char embs created of shape {char_embs.shape}")
    model = StringAutoEncoder(max_char_size=args.max_char_size,
                              hidden_size=args.hidden_size,
                              char_embs=char_embs,
                              dp=args.dp,
                              activate=args.activate,
                              norm=args.norm)

    if args.use_cuda:
        model = send_to_cuda(args.device, model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)

    best_model = deepcopy(model)
    best_top1 = 10 ** 6
    train_loss = 100

    for epoch in range(args.num_epochs):
        if epoch % 20 == 0:
            plot_tsne = True
        else:
            plot_tsne = False

        if epoch % 5 == 0:
            logger.info("validating")
            valid_loss, results = validator.validate(model,  plot_tsne=plot_tsne, epoch=epoch)
            top1 = results[0]
            if top1 > best_top1:
                best_top1 = top1
                best_model = deepcopy(model)
            logger.info('EPOCH - {}, TRAIN LOSS - {:.4f}, VALID LOSS - {:.5f}, Top1:{}, Top10:{}, Top100:{}'
                  .format(epoch, train_loss, valid_loss, results[0], results[1], results[2]))

        logger.info("training")
        train_loss = train_epoch(model, optimizer, train_arr, args)

    save_checkpoint({
        'state_dict': best_model.state_dict(),
        'optimizer': optimizer.state_dict()}, filename=join(model_dir, 'best_model.ckpt'))


if __name__ == '__main__':
    Args, Logger, Model_dir = parse_args()
    Validator, Char_dict, Train_arr = setup(args=Args, logger=Logger, model_dir=Model_dir)
    Logger.info("Starting training.....")

    train(args=Args,
          validator=Validator,
          logger=Logger,
          char_dict=Char_dict,
          train_arr=Train_arr,
          model_dir=Model_dir)

    # for lr in [0.01, 0.05, 0.001, 0.005]:
    #     for wd in [10**-4, 10**-5, 10**-5]:
    #         for dp in [0.1, 0.2, 0.3, 0.5, 0]:
    #             for norm in [True, False]:
    #                 for activate in ['relu', 'sigmoid', 'tanh', '']:
    #                     Args.lr = lr
    #                     Args.wd = wd
    #                     Args.dp = dp
    #                     Args.norm = norm
    #                     Args.activate = activate
    #                     logger.info("Using ")
    #
