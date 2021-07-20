import numpy as np

from nn.activation.abstract import AbsActivation
from nn.interface import IOperator


class Tanh(AbsActivation):

    def __init__(self, op: IOperator = None):
        super().__init__(op)
        self.__ref_output = None

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x, training=True):
        self.__ref_output = np.tanh(x)
        return self.__ref_output

    def do_backward(self, x, grad):
        return np.multiply(1 - np.square(self.__ref_output), grad)
