# This module implements a trainer to be used by train.py

class Trainer(object):

    def __init__(self, loader):
        self.loader = loader

    def _get_next_batch(self):
