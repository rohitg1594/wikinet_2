# Util functions for evaluation
import numpy as np

import torch
from torch.autograd import Variable

from collections import OrderedDict

from src.utils import normalize, relu, equalize_len

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
        context_batch = context_matr[batch_no * batch_size: (batch_no + 1) * batch_size]
        dot_batch = dot_products[batch_no * batch_size: (batch_no + 1) * batch_size]

        C = context_batch.shape[0]
        E = ent_embs.shape[0]

        context_expand = context_batch[:, None, :].repeat(E, axis=1)
        ent_expand = ent_embs[None, :, :].repeat(C, axis=0)
        dot_expand = dot_batch[:, :, None]
        print(context_expand.shape, ent_expand.shape, dot_expand.shape)

        input_vec = np.concatenate((context_expand, dot_expand, ent_expand), axis=2)
        print(input_vec[:5])
        print("Input vec shape : {}".format(input_vec.shape))
        out_hidden = relu(input_vec @ hidden_W.T + hidden_b)
        print(out_hidden[:5])
        print("Out hidden shape : {}".format(out_hidden.shape))

        scores = out_hidden @ output_W.T + output_b
        print(scores[:5])
        print("Scores shape : {}".format(scores.shape))

        preds = np.argmax(scores, axis=2)
        print(preds[:5])
        print("Predictions shape : {}".format(preds.shape))


def full_validation_2(model, dev_data, args, yamda_model):
    model = model.eval()

    context_list = []
    labels_list_list = []
    mask_list = []
    total_correct = 0
    total_mention = 0
    ent_dict = yamda_model['ent_dict']
    num_ents = yamda_model['ent_emb'].shape[0]
    batch_size = 4

    for word_ids, examples in dev_data:
        padded_word_ids = equalize_len(word_ids, args.max_context_size)
        context_list.append(padded_word_ids)

        mask = np.zeros(args.max_ent_size, dtype=np.float32)
        mask[:len(examples)] = 1
        mask_list.append(mask)

        labels_list = [ent_dict[cands[0]]for mention, cands in examples]
        padded_labels = equalize_len(labels_list, args.max_ent_size)
        labels_list_list.append(padded_labels)

    context_tensor = Variable(torch.from_numpy(np.vstack(context_list)))
    print("Context tensor shape: {}".format(context_tensor.shape))

    cand_matrix = np.arange(num_ents)[None, :].repeat(args.max_ent_size, axis=0)
    cand_expand = cand_matrix[None, :, :].repeat(batch_size, axis=0)
    cand_tensor = Variable(torch.from_numpy(cand_expand))
    print("Cand tensor shape: {}".format(cand_tensor.shape))

    if args.use_cuda and isinstance(args.device, int):
        context_tensor = context_tensor.cuda(args.device)
        cand_tensor = cand_tensor.cuda(args.device)

    labels_matr = np.vstack(labels_list_list)
    mask_matr = np.vstack(mask_list)

    print("labels shape: {}".format(labels_matr.shape))
    print("mask shape: {}".format(mask_matr.shape))

    num_batches = context_tensor.shape[0] // batch_size
    for batch_no in range(num_batches):
        print('batch no', batch_no)
        context_batch = context_tensor[batch_no * batch_size: (batch_no + 1) * batch_size, :]
        print('context batch shape', context_batch.shape)
        mask_batch = mask_matr[batch_no * batch_size: (batch_no + 1) * batch_size, :]
        print('mask batch shape', mask_batch.shape)
        mask_batch = mask_batch.reshape(batch_size * args.max_ent_size)
        print('mask batch shape', mask_batch.shape)
        labels_batch = labels_matr[batch_no * batch_size: (batch_no + 1) * batch_size, :]
        print('labels batch shape', labels_batch.shape)
        labels_batch = labels_batch.reshape(batch_size * args.max_ent_size)
        print('labels batch shape', labels_batch.shape)

        scores = model((context_batch, cand_tensor))
        scores = scores.cpu().data.numpy()

        preds = np.argmax(scores, axis=1)
        correct = (np.equal(preds, labels_batch) * mask_batch).sum()
        mention = mask_batch.sum()
        total_correct += correct
        total_mention += mention

    return total_correct, total_mention
