import gensim
import sys
import logging
sys.path.append('/home/rogupta/wikinet_2/')

from src.utils.data import *
from src.utils.utils import *

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


tokenizer = RegexpTokenizer()
WIKI_DIR = '/home/rogupta/enwiki-latest-wikiextractor-2/'
DATA_PATH = '../data/'
NUM_WORKERS = 20
EMB_SIZE = 300

logger.info("Loading Training data.....")
train_data = pickle_load('../data/w2v/training.pickle')
logger.info("Training data loaded.")

logger.info("Tokenizing Training data.....")
tokenized_data = [tokenizer.tokenize(abst) for abst in train_data]
logger.info("Training data tokenized.")

logger.info("Starting Training.....")
model = gensim.models.Word2Vec(tokenized_data, size=EMB_SIZE, workers=NUM_WORKERS, min_count=5, iter=10)
logger.info("Training done.")

logger.info("Saving Model.....")
model.save(os.path.join(DATA_PATH, 'w2v', 'model'))
logger.info("Model Saved.")