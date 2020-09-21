import numpy as np

from nn.interface import IOperator
from nn.abstract import AbsFlexibleUnaryNode
from nn.operation.abstract import OperandHelper


class Square(AbsFlexibleUnaryNode, OperandHelper):

    def __init__(self, op: IOperator):
        super().__init__(op)

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x):
        return np.square(x)

    def do_backward(self, grad):
        return 2 * self.input_ref * grad


class Power(AbsFlexibleUnaryNode, OperandHelper):

    def __init__(self, op: IOperator, power:int):
        super().__init__(op)
        self.__power = power

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x):
        return np.power(x, self.__power)

    def do_backward(self, grad):
        return self.__power * np.power(self.input_ref, self.__power - 1) * grad
