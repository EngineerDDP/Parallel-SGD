import numpy as np

from nn.activation.abstract import AbsActivation
from nn.interface import IOperator


class ReLU(AbsActivation):

    def __init__(self, op: IOperator = None):
        super().__init__(op)
        self.__ref_input: [np.ndarray] = None

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x, training=True):
        self.__ref_input = x.copy
        self.__ref_input[self.__ref_input < 0] = 0
        return self.__ref_input

    def do_backward(self, x, grad):
        return np.multiply(grad, self.__ref_input >= 0)


class LeakReLU(AbsActivation):

    def __init__(self, leak_coefficient=0.1, op: IOperator = None):
        super().__init__(op)
        self.__leak_coef: float = leak_coefficient
        self.__ref_input: np.ndarray

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x, training=True):
        self.__ref_input = x.copy()
        self.__ref_input[self.__ref_input < 0] *= self.__leak_coef
        return self.__ref_input

    def do_backward(self, x, grad):
        r = self.__ref_input.copy()
        r[r > 0] = 1
        r[r < 0] = self.__leak_coef
        return np.multiply(grad, r)
