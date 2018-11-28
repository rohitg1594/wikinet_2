# Script to parse wikipedia dump and generate entity count dictionary
import base64
import glob
import json
import pickle
import logging as logging_master
import re
import argparse
from collections import Counter

import sys
sys.path.extend('/home/rogupta/wikinet_2')
import os
from os.path import join

import findspark
import pyspark

parser = argparse.ArgumentParser(description='process wikipedia dump to generate enitiy count dictionary',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--wiki_dir', type=str, help='Name of the wikipedia file')
parser.add_argument('--data_path', type=str, help='Named Entity Counts model path')
parser.add_argument('--spark_home', type=str, help='Spark home dir')
# parser.add_argument('--resume', action='store_true', help='Continue training model')
args = parser.parse_args()
findspark.init(args.spark_home)


logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('count link structure of wikipedia')
logging.setLevel(logging_master.INFO)

DATA_PATH = args.data_path


def pickle_load(path):
    assert os.path.exists(path)

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


sc = pyspark.SparkContext(appName="wikinetNECOUNTS")
sc.setLogLevel('ERROR')

try:
    logging.info('loading redirects dict')
    with open(os.path.join(DATA_PATH, "redirects.pickle"), 'rb') as f:
        redirects = pickle.load(f)
    logging.info('redirects dict loaded')
except IOError:
    logging.info('redirects not found at {}'.format(os.path.join(DATA_PATH, "redirects_en_2.pickle")))
    redirects = dict()

# try:
#     logging.info("Loading Yamada model.")
#     yamada_model = pickle_load(join(args.data_path, 'yamada', 'yamada_model.pickle'))
#     logging.info("Model loaded.")
#     ent_dict = yamada_model['ent_dict']
# except IOError:
#     logging.info('Error loading yamada model')
#     sys.exit(1)


def uppercase_first(s):
    return s[:1].upper() + s[1:]


def process_line(line):

    wiki_article = json.loads(line)
    wiki_text = ''.join(wiki_article['text']).replace('\n', ' ')

    out = []
    # reversing base64.b64encode(pickle.dumps(self.internal_links)).decode('utf-8')
    for span, mention_page_name in sorted(
            pickle.loads(base64.b64decode(wiki_article['internal_links'].encode('utf-8'))).items(),
            key=lambda x: x[0][0]):

        if len(mention_page_name) < 2:
            continue
        begin, end = span
        if begin == end:
            continue

        mention, page_name = mention_page_name[:2]
        page_name = uppercase_first(page_name.replace(' ', '_'))
        if page_name in redirects:
            page_name = redirects[page_name]

        if 'Category:' in page_name:
            continue

        out.append((wiki_text[begin:end], page_name))
    return out


def counter_update(c, v):
    return c.update(v)


if __name__ == "__main__":

    RE_FILE_MATCH = re.compile(r'.*\/wiki_\d\d$')
    f_names = []
    for f_name in glob.glob(args.wiki_dir + '/**', recursive=True):
        if RE_FILE_MATCH.match(f_name):
            f_names.append(f_name)
    logging.info('{} files to be processed'.format(len(f_names)))

    logging.info('creating rdd')
    ne_counts = sc.textFile(','.join(f_names)) \
        .filter(lambda x: len(x) > 1) \
        .map(lambda x: process_line(x)) \
        .flatMap(lambda x: x) \
        .groupByKey() \
        .mapValues(list) \
        .combineByKey(Counter, counter_update, lambda x, y: x + y)
    logging.info('RDD created.')

    logging.info("Size of RDD : {} MB".format(sys.getsizeof(ne_counts) / 10 ** 6))
    ne_counts_dict = ne_counts.collectAsMap()
    logging.info("Dict created.")
    logging.info("Size of dictionary : {} MB".format(sys.getsizeof(ne_counts_dict) / 10 ** 6))

    with open(join(args.data_path, 'necounts', 'new_necounts.pickle'), 'wb') as f:
        pickle.dump(ne_counts_dict, f)
    logging.info("Done.")
