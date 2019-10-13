import numpy as np


class ILoss:

    def __init__(self):
        pass

    def loss(self, y, label):
        raise NotImplementedError()

    def gradient(self, y, label):
        raise NotImplementedError()


class MseLoss:

    def __init__(self):
        pass

    def loss(self, y, label):
        return np.mean(np.square(label - y))

    def gradient(self, y, label):
        return (label - y) * -2


class CrossEntropyLoss:

    def __init__(self):
        pass

    def loss(self, y, label):
        # multiply element-wise
        return np.mean(np.multiply(label, np.log(y)) + np.multiply((1 - label), np.log(1 - y))) * -1

    def gradient(self, y, label):
        return ((1 - label) / (1 - y) - label / y) / label.shape[1]


class CrossEntropyLossWithSigmoid:

    def __init__(self):
        pass

    def loss(self, y, label):
        # multiply element-wise
        return np.mean(np.multiply(label, np.log(y)) + np.multiply((1 - label), np.log(1 - y))) * -1

    def gradient(self, y, label):
        return np.multiply(y - label, 1 / np.multiply(y, 1 - y))


class TanhLoss:

    def __init__(self):
        pass

    def loss(self, y, label):
        return np.mean(np.square(np.tanh(label - y)))

    def gradient(self, y, label):
        return -2.0 * np.multiply(np.tanh(label - y), (1 - np.square(np.tanh(label - y))))


class L1Loss:

    def __init__(self):
        self.L2Loss = MseLoss()

    def loss(self, y, label):
        return np.mean(label - y)

    def gradient(self, y, label):

        grad = label - y
        grad[np.where(grad < 0)] = -1
        grad[np.where(grad > 0)] = 1
        grad = grad * -1 + self.L2Loss.gradient(y, label)

        return grad
