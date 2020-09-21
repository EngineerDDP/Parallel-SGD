import numpy as np

from nn.interface import IValue
from nn.gradient_descent.interface import IGradientDescent


class RMSPropOptimizer(IGradientDescent):

    def __init__(self, learning_rate=0.01):
        self.__eta = learning_rate
        self.__r = 0
        self.__epsilon = 1e-7
        self.__alpha = 0.9

    def delta(self, var: IValue):
        self.__r = (1 - self.__alpha) * np.square(var) + self.__alpha * self.__r
        return self.__eta * var / (self.__epsilon + np.sqrt(self.__r))

    def __str__(self):
        return "<RMSProp Optimizer>"
