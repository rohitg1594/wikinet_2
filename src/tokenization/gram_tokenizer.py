# Functions for various types of gram tokenizers


def get_gram_tokenizer(gram_type=None):
    if gram_type == 'unigram':
        return unigram_tokenizer
    elif gram_type == 'bigram':
        return  bigram_tokenizer
    elif gram_type == 'trigram':
        return trigram_tokenizer
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
    if len(x) <= 3:
        return x
    else:
        return [b[0] + b[1] + b[2] for b in zip(x[:-2], x[1:-1], x[2:])]