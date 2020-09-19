import numpy as np

from nn.interface import IOperator, IValue
from nn.operation.abstract import AbsBinaryOperator, AbsUnaryOperator


def dfs_search(op):
    if isinstance(op, IValue):
        return {op.id}
    if isinstance(op, AbsUnaryOperator):
        return dfs_search(op.op_child)

    if isinstance(op.op_left, AbsBinaryOperator):
        left = dfs_search(op.op_left)
    elif isinstance(op.op_left, IValue):
        left = {op.op_left.id}
    else:
        left = set()

    if isinstance(op.op_right, AbsBinaryOperator):
        right = dfs_search(op.op_right)
    elif isinstance(op.op_right, IValue):
        right = {op.op_right.id}
    else:
        right = set()

    return left | right


class Multiply(AbsBinaryOperator):

    def __init__(self, op1:IOperator, op2:IOperator):
        super().__init__(op1, op2)
        assert len(dfs_search(op1) & dfs_search(op2)) == 0, "Cannot multiply to elements with same child node."
        # check shape
        self.output_shape()

    def output_shape(self) -> [list, tuple, None]:
        if self.op_left.output_shape() is None or self.op_right.output_shape() is None:
            return None

        if len(self.op_left.output_shape()) != 2 or len(self.op_right.output_shape()) != 2\
            or self.op_left.output_shape()[1] != self.op_right.output_shape()[0]:
            raise AssertionError("Operation cannot brand between {} and {}".format(self.op_left.output_shape(), self.op_right.output_shape()))

        return [self.op_left.output_shape()[0], self.op_right.output_shape()[1]]

    def do_forward(self, arg1, arg2):
        return np.dot(arg1, arg2)

    def do_backward(self, grad):
        return np.dot(grad, self.right_input.T), np.dot(self.left_input.T, grad)
