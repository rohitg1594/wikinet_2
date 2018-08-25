# Tokenizer used for words by yamada
import re


class RegexpTokenizer(object):
    __slots__ = ('_rule',)

    def __init__(self, rule=r"[\w\d]+"):
        self._rule = re.compile(rule, re.UNICODE)

    def tokenize(self, text):
        all_capitalized = True
        for c in text:
            if not c.isupper():
                all_capitalized = False
        if all_capitalized:
            text = text.title()
        return [text[o.start():o.end()] for o in self._rule.finditer(text)]
