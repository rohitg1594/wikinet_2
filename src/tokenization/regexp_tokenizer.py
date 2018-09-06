# Tokenizer used for words by yamada
import re


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