# Logging Module
from os.path import join
import os
import sys

from datetime import datetime

import logging


def get_logger(args):
    # Logging
    logger = logging.getLogger()
    log_formatter = logging.Formatter(fmt='%(levelname)s:%(asctime)s:%(message)s', datefmt='%I:%M:%S %p')

    log_path = join(args.data_path, 'logs', '{}_{}.log'.format(datetime.now().strftime("%Y_%m_%d"), args.exp_name))
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler, )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.level = 10