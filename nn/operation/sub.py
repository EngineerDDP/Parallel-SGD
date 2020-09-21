from nn.interface import IOperator
from nn.operation.abstract import AbsBinaryOperator


class Sub(AbsBinaryOperator):

    def __init__(self, op1:IOperator, op2:IOperator):
        super().__init__(op1, op2)

    def output_shape(self) -> [list, tuple, None]:
        return self.op_left.output_shape()

    def do_forward(self, arg1, arg2):
        return arg1 - arg2

    def do_backward(self, grad):
        return grad, - grad
