from nn.interface import IOperator
from nn.operation.abstract import AbsBinaryOperator


class Conv2D(AbsBinaryOperator):

    def __init__(self, op1, op2, padding, strike):
        super().__init__(op1, op2)

    def do_forward(self, arg1, arg2):
        pass

    def do_backward(self, grad):
        pass

    def output_shape(self) -> [list, tuple, None]:
        pass