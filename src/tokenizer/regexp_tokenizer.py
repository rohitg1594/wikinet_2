# Tokenizer used for words by yamada
import re


class Token:
    __slots__ = ('text', 'span')

    def __init__(self, text, span):
        self.text = text
        self.span = span

    def __repr__(self):
        return '<Token %s>' % self.text.encode('utf-8')

    def __reduce__(self):
        return self.__class__, (self.text, self.span)


class RegexpTokenizer(object):
    __slots__ = ('_rule', 'lower')

    def __init__(self, rule=r"[\w\d]+", lower=True):
        self._rule = re.compile(rule, re.UNICODE)
        self.lower = lower

    def tokenize(self, text):
        if self.lower:
            return [Token(text[o.start():o.end()].lower(), o.span()) for o in self._rule.finditer(text)]
        else:
            return [Token(text[o.start():o.end()], o.span()) for o in self._rule.finditer(text)]
