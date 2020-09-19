import numpy as np

from nn.interface import IOperator
from nn.operation.abstract import AbsUnaryOperator
from nn.operation.interface import IActivation


class Softmax_NoGradient(AbsUnaryOperator, IActivation):

    def __init__(self, op: IOperator=None):
        super().__init__(op)

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x):
        denominator = np.sum(np.exp(x), axis=1).reshape([-1, 1])
        return np.exp(x) / denominator

    def do_backward(self, grad):
        return grad
