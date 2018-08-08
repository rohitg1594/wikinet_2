# Util functions for evaluation
import numpy as np

def eval_ranking(I, gold, ks, also_topk=True):
    if also_topk:
        topks = np.zeros(len(ks))
        for j, k in enumerate(ks):
            for i in range(I.shape[0]):
                if gold[i] in I[i, :k]:
                    topks[j] += 1

        topks /= I.shape[0]

        for k, topk in zip(ks, topks):
            print('Top {} precision : {:.7f}'.format(k, topk))

    # Mean Reciprocal Rank
    ranks = []
    for i in range(I.shape[0]):
        index = np.where(gold[i] == I[i])[0] + 1
        if not index:
            ranks.append(1 / I.shape[1])
        else:
            ranks.append(1 / index)
    mrr = np.mean(np.array(ranks))
    # print('Mean Reciprocal Rank : {}'.format(mrr))
    if not isinstance(mrr, float):
        mrr = mrr[0]

    return topks[0], topks[1], topks[2], mrr