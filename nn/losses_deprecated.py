import numpy as np

from nn.interface import ILoss


class MseLoss(ILoss):
    """
        Mse loss.
    """

    def metric(self, y, label):
        return np.mean(np.square(label - y))

    def gradient(self, y, label):
        return (label - y) * -2


class CrossEntropyLoss(ILoss):

    def metric(self, y, label):
        # multiply element-wise
        return np.mean(np.multiply(label, np.log(y)) + np.multiply((1 - label), np.log(1 - y))) * -1

    def gradient(self, y, label):
        return (1 - label) / (1 - y) - label / y


class CrossEntropyLossWithSigmoid(CrossEntropyLoss):

    def gradient(self, y, label):
        return np.multiply(y - label, 1 / np.multiply(y, 1 - y))


class CrossEntropyLossWithSoftmax(CrossEntropyLoss):

    def gradient(self, y, label):
        return y - label


class TanhLoss(ILoss):

    def metric(self, y, label):
        return np.mean(np.square(np.tanh(label - y)))

    def gradient(self, y, label):
        return -2.0 * np.multiply(np.tanh(label - y), (1 - np.square(np.tanh(label - y))))


class L1Loss(ILoss):

    def __init__(self):
        self.L2Loss = MseLoss()

    def metric(self, y, label):
        return np.mean(label - y)

    def gradient(self, y, label):

        grad = label - y
        grad[np.where(grad < 0)] = -1
        grad[np.where(grad > 0)] = 1
        grad = grad * -1 + self.L2Loss.gradient(y, label)

        return grad
