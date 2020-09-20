import numpy as np

from nn.interface import IOperator
from nn.operation.abstract import AbsUnaryOperator
from nn.operation.interface import IActivation


class ReLU(AbsUnaryOperator, IActivation):

    def __init__(self, op: IOperator=None):
        super().__init__(op)

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x):
        r = x.copy()
        r[r < 0] = 0
        return r

    def do_backward(self, grad):
        return np.multiply(grad, self.input_ref >= 0)


class LeakReLU(AbsUnaryOperator, IActivation):

    def __init__(self, leak_coefficient=0.1, op: IOperator=None):
        super().__init__(op)
        self.__leak_coef = leak_coefficient

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x):
        r = x.copy()
        r[r < 0] *= self.__leak_coef
        return r

    def do_backward(self, grad):
        r = self.output_ref.copy()
        r[r > 0] = 1
        r[r < 0] = self.__leak_coef
        return np.multiply(grad, r)
