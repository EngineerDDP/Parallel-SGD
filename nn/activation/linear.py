import numpy as np
from numpy import ndarray

from nn.interface import IOperator
from nn.activation.abstract import AbsActivation


class Linear(AbsActivation):

    def do_forward(self, x, training=True):
        return x

    def do_backward(self, x, grad):
        return grad


class HTanh(AbsActivation):

    def __init__(self, op: IOperator = None, leak_coefficient: float = 1e-2):
        super().__init__(op)
        self.__leak_coefficient = leak_coefficient
        self.__mask: ndarray = np.ones(shape=1)

    def do_forward(self, x: [float, ndarray], training: bool = True) -> [float, ndarray]:
        self.__mask = np.multiply(-1 < x, x < 1)
        self.__mask[self.__mask == 0] = self.__leak_coefficient
        return np.multiply(x, self.__mask)

    def do_backward(self, x: [float, ndarray], grad: [float, ndarray]) -> [ndarray, float]:
        return np.multiply(grad, self.__mask)

    def clear_unused(self):
        self.__mask = np.ones(shape=1)
