import gensim
import sys
import argparse
sys.path.append('/home/rogupta/wikinet_2/')

from src.utils.data import *
from src.utils.utils import *

parser = argparse.ArgumentParser(description='Training Gensim')
parser.add_argument('--data_path', required=True, type=str, help='config file path')
parser.add_argument('--num_epochs', required=True, type=int, help='number of epochs')
parser.add_argument('--num_workers', required=True, type=int, help='number of workers')
parser.add_argument('--emb_size', required=True, type=int, help='embedding size')
parser.add_argument('--min_count', required=True, type=int, help='minimum count considered while training')
parser.add_argument('--train_type', required=True, type=str, help='type of training data - new or yamada')

logger = logging.getLogger()
log_formatter = logging.Formatter(fmt='%(levelname)s:%(asctime)s:%(message)s', datefmt='%I:%M:%S %p')
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logger.level = 10


class RegexpTokenizer(object):
    __slots__ = ('_rule', 'lower')

    def __init__(self, rule=r"[\w\d]+", lower=True):
        self._rule = re.compile(rule, re.UNICODE)
        self.lower = lower

    def tokenize(self, text):
        if self.lower:
            return [text[o.start():o.end()].lower() for o in self._rule.finditer(text)]
        else:
            return [text[o.start():o.end()] for o in self._rule.finditer(text)]


args = parser.parse_args()
tokenizer = RegexpTokenizer()
WIKI_DIR = '/work/rogupta/enwiki-latest-wikiextractor-2/'

logger.info("Loading Training data.....")
train_data = pickle_load(join(args.data_path, f'w2v/train-{args.train_type}-dict.pickle'))
logger.info("Training data loaded.")

logger.info("Tokenizing Training data.....")
tokenized_data = [tokenizer.tokenize(abst) for abst in train_data]
logger.info("Training data tokenized.")

logger.info("Starting Training.....")
model = gensim.models.Word2Vec(tokenized_data,
                               size=args.emb_size,
                               workers=args.num_workers,
                               min_count=args.min_count,
                               iter=args.num_epochs)
logger.info("Training done.")

logger.info("Saving Model.....")
output_dir = os.path.join(args.data_path, 'w2v', f'w2v-{args.train_type}-{args.emb_size}-{args.num_epochs}-{args.min_count}')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model.save(join(output_dir, 'model'))
logger.info("Model Saved.")
