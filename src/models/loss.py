# Loss class that models will inherit
import torch.nn.functional as F


class Loss:

    @staticmethod
    def cross_entropy(scores, labels):
        loss = F.cross_entropy(scores, labels)
        loss = loss.sum()

        return loss

    @staticmethod
    def binary_cross_entropy(scores, labels):
        loss = F.binary_cross_entropy(scores, labels)

        return loss