import numpy as np

from nn.abstract import AbsFlexibleUnaryNode
from nn.interface import IOperator
from nn.operation.abstract import OperandHelper


class Square(AbsFlexibleUnaryNode, OperandHelper):

    def __init__(self, op: IOperator):
        super().__init__(op)

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x: [float, np.ndarray], training: bool = True) -> np.ndarray:
        return np.square(x)

    def do_backward(self, x: [float, np.ndarray], grad: [float, np.ndarray]) -> [np.ndarray, float]:
        return 2 * x * grad


class Power(AbsFlexibleUnaryNode, OperandHelper):

    def __init__(self, op: IOperator, power:int):
        super().__init__(op)
        self.__power = power

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x: [float, np.ndarray], training: bool = True) -> np.ndarray:
        return np.power(x, self.__power)

    def do_backward(self, x: [float, np.ndarray], grad: [float, np.ndarray]) -> np.ndarray:
        return self.__power * np.power(x, self.__power - 1) * grad
