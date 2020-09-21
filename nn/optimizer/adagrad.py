import numpy as np

from nn.interface import IValue
from nn.optimizer.interface import IGradientDescent


class AdaGradOptimizer(IGradientDescent):

    def __init__(self, learning_rate=0.01):
        self.__eta = learning_rate
        self.__r = 0
        self.__epsilon = 1e-7

    def delta(self, var: IValue):
        self.__r = np.square(var) + self.__r
        return self.__eta * var / (self.__epsilon + np.sqrt(self.__r))

    def __str__(self):
        return "<AdaGrad Optimizer>"
