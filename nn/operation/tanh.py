import numpy as np

from nn.interface import IOperator
from nn.operation.abstract import AbsUnaryOperator
from nn.operation.interface import IActivation


class Tanh(AbsUnaryOperator, IActivation):

    def __init__(self, op: IOperator=None):
        super().__init__(op)

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x):
        return np.tanh(x)

    def do_backward(self, grad):
        return 1 - np.square(self.output_ref)
