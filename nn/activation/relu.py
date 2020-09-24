import numpy as np

from nn.activation.abstract import AbsActivation
from nn.interface import IOperator


class ReLU(AbsActivation):

    def __init__(self, op: IOperator=None):
        super().__init__(op)

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x, training=True):
        r = x.copy()
        r[r < 0] = 0
        return r

    def do_backward(self, x, grad):
        return np.multiply(grad, self.input_ref >= 0)


class LeakReLU(AbsActivation):

    def __init__(self, leak_coefficient=0.1, op: IOperator=None):
        super().__init__(op)
        self.__leak_coef = leak_coefficient

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x, training=True):
        r = x.copy()
        r[r < 0] *= self.__leak_coef
        return r

    def do_backward(self, x, grad):
        r = self.output_ref.copy()
        r[r > 0] = 1
        r[r < 0] = self.__leak_coef
        return np.multiply(grad, r)
