import numpy as np

from nn.interface import IOperator
from nn.operation.abstract import AbsUnaryOperator
from nn.operation.interface import IActivation


class Sigmoid(AbsUnaryOperator, IActivation):

    def __init__(self, delta=0.0, op: IOperator=None):
        super().__init__(op)
        self.__delta = delta

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x):
        return 1 / (1 + np.exp(-1 * (x + self.__delta)))

    def do_backward(self, grad):
        return np.multiply(self.output_ref, (1 - self.output_ref))


class SigmoidNoGrad(Sigmoid):

    def __init__(self, delta=0.0, op: IOperator=None):
        super().__init__(delta, op)

    def do_backward(self, grad):
        return grad
