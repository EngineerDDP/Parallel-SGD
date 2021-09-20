import numpy as np

from nn.loss.abstract import ILoss


class TanhLoss(ILoss):

    def __init__(self):
        pass

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<L2 Tanh loss>"

    def metric(self, arg1, arg2):
        return np.mean(np.square(np.tanh(arg2 - arg1)))

    def gradient(self, arg1, arg2):
        tmp_tanh = np.tanh(arg2 - arg1)
        return -2.0 * np.multiply(tmp_tanh, (1 - np.square(tmp_tanh))), 2.0 * np.multiply(tmp_tanh, (1 + np.square(tmp_tanh)))
