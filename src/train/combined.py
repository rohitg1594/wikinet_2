# Main training file
from datetime import datetime
import configargparse

from src.utils.utils import *
from src.utils.dictionary import Dictionary  # needed because of autoencoder
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
    padding.add_argument('--max_char_size', type=int, help='max number of grams')
    padding.add_argument('--max_ent_size', type=int, help='max number of entities considered in abstract')
    padding.add_argument('--ignore_init', type=str2bool, help='whether to ignore first five tokens of context')

    # Model Type
    model_selection = parser.add_argument_group('Type of model to train.')
    model_selection.add_argument('--model_name', type=str, help='type of model to train')

    # Embeddings
    model_embs = parser.add_argument_group('Different embedding types')
    model_embs.add_argument('--init_gram_embs', type=str, help="initialize gram embeddings")
    model_embs.add_argument('--init_context_embs', type=str, help="initialize context embeddings")
    model_embs.add_argument('--init_mention_embs', type=str, help="initialize mention embeddings")
    model_embs.add_argument('--init_char_embs', type=str, help="initialize char embeddings")
    model_embs.add_argument('--mention_word_dim', type=int, help='dimension of mention word embeddings')
    model_embs.add_argument('--context_word_dim', type=int, help='dimension of mention word embeddings')
    model_embs.add_argument('--mention_ent_dim', type=int, help='dimension of mention entity embeddings')

    # Model params
    model_params = parser.add_argument_group("Parameters for chosen model.")
    model_params.add_argument('--measure', type=str, default='ip', choices=['ip', 'l2'], help='faiss index')
    model_params.add_argument('--dp', type=float, help='drop out')
    model_params.add_argument('--activate', type=str, help='activation function after dropout')
    model_params.add_argument('--sigmoid', type=str2bool, help='whether to sigmoid the weights for linear_scalar model')
    model_params.add_argument('--init_stdv', type=float,
                              help='standard deviation to initialize embeddings in small context')
    model_params.add_argument('--combined_linear', type=str2bool,
                              help='whether to have a combining linear layer in small context model')
    model_params.add_argument('--init_linear', type=float, help='initialize linear layers')

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
    train_params.add_argument('--optim', type=str, choices=['adagrad', 'adam', 'rmsprop', 'sparseadam'], help='optimizer')

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
    if args.optim == 'sparseadam':
        args.sparse = True
    else:
        args.sparse = False
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
    logger.info("Loading Yamada model.....")
    yamada_model = pickle_load(join(args.data_path, 'yamada', args.yamada_model))
    logger.info("Model loaded.")

    # Gram Embeddings
    gram_tokenizer = get_gram_tokenizer(gram_type=args.gram_type, lower_case=args.gram_lower)
    logger.info(f"Using gram tokenizer {gram_tokenizer.__name__}")
    gram_dict = load_vocab(join(args.data_path, 'gram_vocabs', args.gram_type + '.tsv'), plus_one=True)
    gram_embs = normal_initialize(len(gram_dict) + 1, args.gram_dim)
    logger.info(f"Gram embeddings created of shape: {gram_embs.shape}")

    # Char Embeddings for autoencoder
    logger.info(f'Loading char embeddings from autoencoder state dict {args.init_char_embs}.....')
    autoencoder_state_dict = torch.load(args.init_char_embs, map_location='cpu')['state_dict']
    char_embs = autoencoder_state_dict['char_embs.weight']
    hidden_size = autoencoder_state_dict['lin2.weight'].shape[0]
    logger.info(f'Char embeddings loaded')

    # Context Embeddings
    word_embs, ent_embs, W, b = get_context_embs(args.data_path, args.init_context_embs, yamada_model)
    logger.info(f'Context embeddings loaded, word_embs : {word_embs.shape}, ent_embs : {ent_embs.shape}')
    if args.model_name == 'full_context_string_from_scratch_ent':
        ent_embs = normal_initialize(ent_embs.shape[0], args.mention_word_dim + args.context_word_dim + hidden_size)

    # Mention Embeddings
    logger.info("Loading mention embeddings.....")
    mention_word_embs, mention_ent_embs = get_mention_embs(args.init_mention_embs,
                                                           num_word=word_embs.shape[0],
                                                           mention_word_dim=args.mention_word_dim,
                                                           num_ent=ent_embs.shape[0],
                                                           mention_ent_dim=args.mention_ent_dim)
    logger.info(f'Mention embeddings loaded, mention_word_embs : {mention_word_embs.shape},'
                f' mention_ent_embs : {mention_ent_embs.shape}')

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
                                  word2id=yamada_model['word_dict'],
                                  ent2id=yamada_model['ent_dict'],
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

    return {'train_loader':train_loader,
            'validator': validator,
            'W': W,
            'b': b,
            'ent_embs': ent_embs,
            'word_embs': word_embs,
            'mention_word_embs': mention_word_embs,
            'mention_ent_embs': mention_ent_embs,
            'gram_embs': gram_embs,
            'char_embs': char_embs,
            'hidden_size': hidden_size,
            'autoencoder_state_dict': autoencoder_state_dict}


def train(**kwargs):

    # Unpack args
    logger = kwargs['logger']
    validator = kwargs['validator']
    model_dir = kwargs['model_dir']
    train_loader = kwargs['train_loader']
    args = kwargs['args']

    # Model
    model = get_model(**kwargs)

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
    setup_dict = setup(args=Args, logger=Logger)
    setup_dict['logger'] = Logger
    setup_dict['args'] = Args
    setup_dict['model_dir'] = Model_dir

    train(**setup_dict)
