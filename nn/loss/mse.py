import numpy as np

from nn.interface import IOperator
from nn.loss.abstract import ILoss


class MSELoss(ILoss):

    def __init__(self):
        pass

    def __repr__(self):
        print(self.__str__())

    def __str__(self):
        return "<Mean Square Error Loss>"

    def gradient(self, arg1, arg2):
        return 2.0 * (arg1 - arg2), -2.0 * (arg1 - arg2)

    def metric(self, arg1, arg2):
        return np.sum(np.square(arg1 - arg2))
