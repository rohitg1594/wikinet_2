# Functions for various types of gram tokenizers


def get_gram_tokenizer(gram_type=None, lower_case=False):
    if not lower_case:
        if gram_type == 'unigram':
            return unigram_tokenizer
        elif gram_type == 'bigram':
            return bigram_tokenizer
        elif gram_type == 'trigram':
            return trigram_tokenizer
        else:
            print("Token type {} not implemented".format(gram_type))
            raise NotImplementedError
    if lower_case:
        if gram_type == 'unigram':
            return lower_unigram_tokenizer
        elif gram_type == 'bigram':
            return lower_bigram_tokenizer
        elif gram_type == 'trigram':
            return lower_trigram_tokenizer
        else:
            print("Token type {} not implemented".format(gram_type))
            raise NotImplementedError


def unigram_tokenizer(x):
    return list(x)


def bigram_tokenizer(x):
    if len(x) <= 2:
        return x
    else:
        return [b[0] + b[1] for b in zip(x[:-1], x[1:])]


def trigram_tokenizer(x):
    x = x.lower()
    if len(x) <= 3:
        return x
    else:
        return [b[0] + b[1] + b[2] for b in zip(x[:-2], x[1:-1], x[2:])]


def lower_unigram_tokenizer(x):
    return list(x)


def lower_bigram_tokenizer(x):
    x = x.lower()
    if len(x) <= 2:
        return x
    else:
        return [b[0] + b[1] for b in zip(x[:-1], x[1:])]


def lower_trigram_tokenizer(x):
    x = x.lower()
    if len(x) <= 3:
        return x
    else:
        return [b[0] + b[1] + b[2] for b in zip(x[:-2], x[1:-1], x[2:])]