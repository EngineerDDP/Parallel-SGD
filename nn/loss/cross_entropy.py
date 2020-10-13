import numpy as np
from numpy import ndarray

from nn.loss.abstract import ILoss


class Cross_Entropy(ILoss):

    def __init__(self):
        self.__epsilon = 1e-9

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Cross Entropy Loss>"

    def metric(self, arg1, arg2):
        q = np.multiply(arg2, np.log(arg1 + self.__epsilon))
        p = np.multiply((1 - arg2), np.log(1 - arg1 + self.__epsilon))
        return -1 * np.mean(np.sum(q + p, axis=1))

    def gradient(self, left: [float, ndarray], right: [float, ndarray]) -> (ndarray, ndarray):
        return (1 - right) / (1 - left) - right / left, (1 - left) / (1 - right) - left / right


class Cross_Entropy_With_Softmax(ILoss):

    def __init__(self):
        self.__epsilon = 1e-9

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Cross Entropy Loss>"

    def metric(self, arg1, arg2):
        q = np.multiply(arg2, np.log(arg1 + self.__epsilon))
        p = np.multiply((1 - arg2), np.log(1 - arg1 + self.__epsilon))
        return -1 * np.mean(np.sum(q + p, axis=1))

    def gradient(self, arg1, arg2):
        return arg1 - arg2, arg2 - arg1
