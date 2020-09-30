import numpy as np

from nn.activation.abstract import AbsActivation
from nn.interface import IOperator


class Softmax_NoGradient(AbsActivation):

    def __init__(self, op: IOperator = None):
        super().__init__(op)

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x, training=True):
        x = (x - np.mean(x)) / np.std(x)
        denominator = np.sum(np.exp(x), axis=1).reshape([-1, 1])
        return np.exp(x) / denominator

    def do_backward(self, x, grad):
        return grad
