import numpy as np

from nn.activation.abstract import AbsActivation
from nn.interface import IOperator


class Sigmoid(AbsActivation):

    def __init__(self, delta=0.0, op: IOperator=None):
        super().__init__(op)
        self.__delta = delta
        self.__ref_output = None

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x, training=True):
        self.__ref_output = 1 / (1 + np.exp(-1 * (x + self.__delta)))
        return self.__ref_output

    def do_backward(self, x, grad):
        return np.multiply(self.__ref_output, (1 - self.__ref_output))


class SigmoidNoGrad(Sigmoid):

    def __init__(self, delta=0.0, op: IOperator=None):
        super().__init__(delta, op)

    def do_backward(self, grad):
        return grad
