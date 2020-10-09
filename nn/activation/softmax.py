import numpy as np

from nn.activation.abstract import AbsActivation
from nn.interface import IOperator


class Softmax(AbsActivation):

    def __init__(self, op: IOperator = None):
        super().__init__(op)
        self.__ref_mean = 0
        self.__ref_std = 1e-9
        self.__epsilon = 1e-2

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x, training=True):
        # self.__ref_mean = np.mean(x, axis=0)
        # self.__ref_std = np.std(x, axis=0)
        # x = (x - self.__ref_mean) / (self.__ref_std + self.__epsilon)
        denominator = np.sum(np.exp(x), axis=1).reshape([-1, 1])
        return np.exp(x) / denominator

    def do_backward(self, x, grad):
        return grad
