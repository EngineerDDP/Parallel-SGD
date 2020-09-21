import numpy as np

from nn.interface import IValue
from nn.optimizer.interface import IGradientDescent


class AdaDeltaOptimizer(IGradientDescent):

    def __init__(self):
        self.__r = 0
        self.__epsilon = 1e-6
        self.__alpha = 0.95
        self.__x = 0

    def delta(self, var: IValue):
        self.__r = (1 - self.__alpha) * np.square(var) + self.__alpha * self.__r
        result = (np.sqrt(self.__x) + self.__epsilon) / (np.sqrt(self.__r) + self.__epsilon)
        self.__x = (1 - self.__alpha) * np.square(result) + self.__alpha * self.__x
        result = (np.sqrt(self.__x) + self.__epsilon) / (np.sqrt(self.__r) + self.__epsilon)
        return result

    def __str__(self):
        return "<AdaDelta Optimizer>"
