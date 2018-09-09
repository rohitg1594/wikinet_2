# Util functions for eval
import numpy as np
import random

from collections import defaultdict


def check_errors(I, gold, gram_indices, rev_ent_dict, rev_gram_dict, ks):
    errors = defaultdict(list)

    for j, k in enumerate(ks):
        for i in range(I.shape[0]):
            if gold[i] not in I[i, :k]:
                errors[k].append((i, gold[i], I[i]))

    for k, errors in errors.items():
        print("Top {} errors:".format(k))
        mask = random.sample(range(len(errors)), 10)
        for i in mask:
            mention_idx, gold_id, predictions_id = errors[i]
            mention_tokens = gram_indices[mention_idx]
            predictions = ','.join([rev_ent_dict.get(ent_id, '') for ent_id in predictions_id][:10])

            mention_grams = []
            for token_idx, token in enumerate(mention_tokens):
                if token == 0:
                    break
                elif token in rev_gram_dict:
                    mention_grams.append(rev_gram_dict[token][0])
            mention = ''.join(mention_grams)
            if token_idx > 0:
                last_gram = rev_gram_dict.get(mention_tokens[token_idx-1], '')
                if len(last_gram) > 1:
                    mention += last_gram[1:]

            print('{}|{}|{}'.format(mention, rev_ent_dict.get(gold_id, ''), predictions))
        print()


def eval_ranking(I, gold, ks):
    topks = np.zeros(len(ks))

    for j, k in enumerate(ks):
        for i in range(I.shape[0]):
            if gold[i] in I[i, :k]:
                topks[j] += 1

    topks /= I.shape[0]

    # Mean Reciprocal Rank
    ranks = []
    for i in range(I.shape[0]):
        index = np.where(gold[i] == I[i])[0] + 1
        if not index:
            ranks.append(1 / I.shape[1])
        else:
            ranks.append(1 / index)
    mrr = np.mean(np.array(ranks))

    if not isinstance(mrr, float):
        mrr = mrr[0]

    return topks[0], topks[1], topks[2], mrr
