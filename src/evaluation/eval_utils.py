# Util functions for evaluation
import numpy as np

from collections import OrderedDict

from src.utils import normalize, relu

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


def full_validation(model, dev_data, ent_dict):
    new_state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v

    word_embs = new_state_dict['embeddings_word.weight'].cpu().numpy()
    ent_embs = new_state_dict['embeddings_ent.weight'].cpu().numpy()
    orig_W = new_state_dict['orig_linear.weight'].cpu().numpy()
    orig_b = new_state_dict['orig_linear.bias'].cpu().numpy()
    hidden_W = new_state_dict['hidden.weight'].cpu().numpy()
    hidden_b = new_state_dict['hidden.bias'].cpu().numpy()
    output_W = new_state_dict['output.weight'].cpu().numpy()
    output_b = new_state_dict['output.bias'].cpu().numpy()

    context_list = []
    gold = []
    for word_ids, examples in dev_data:
        context_vec = normalize(word_embs[word_ids].mean(axis=0) @ orig_W + orig_b)
        for mention, cands in examples:
            gold.append(ent_dict[cands[0]])
            context_list.append(context_vec)
    context_matr = np.vstack(context_list)
    gold = np.array(gold)
    print(context_matr[:5])
    print("Shape of context matrix : {}".format(context_matr.shape))
    dot_products = context_matr @ ent_embs.T
    print(dot_products[:5])
    print("Shape of dot products : {}".format(dot_products.shape))
    print(gold[:10])
    print("Shape Gold : {}".format(gold.shape))

    batch_size = 50
    num_batches = context_matr.shape[0] // batch_size
    for batch_no in range(num_batches):
        batch = context_matr[batch_no * batch_size : (batch_no + 1) * batch_size]

        C = batch.shape[0]
        E = ent_embs.shape[0]

        context_expand = batch[:, None, :].repeat(E, axis=1)
        ent_expand = ent_embs[None, :, :].repeat(C, axis=0)
        dot_expand = dot_products[:, :, None]

        input_vec = np.concatenate((context_expand, dot_expand, ent_expand), axis=2)
        print(input_vec[:5])
        print("Input vec shape : {}".format(input_vec.shape))
        out_hidden = relu(input_vec @ hidden_W + hidden_b)
        print(out_hidden[:5])
        print("Out hidden shape : {}".format(out_hidden.shape))

        scores = out_hidden @ output_W + output_b
        print(scores[:5])
        print("Scores shape : {}".format(scores.shape))

        preds = np.argmax(scores, axis=2)
        print(preds[:5])
        print("Predictions shape : {}".format(preds.shape))
