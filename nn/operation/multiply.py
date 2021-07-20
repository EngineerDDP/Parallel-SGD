import numpy as np
from numpy import ndarray

from nn.interface import IOperator, IValue, IBinaryNode, IUnaryNode
from nn.abstract import AbsFlexibleBinaryNode, AbsFlexibleUnaryNode
from nn.operation.abstract import OperandHelper


def dfs_search(op):
    if isinstance(op, IValue):
        return {op.id}
    if isinstance(op, IUnaryNode):
        return dfs_search(op.op_child)
    if isinstance(op, IBinaryNode):
        return dfs_search(op.op_left) | dfs_search(op.op_right)

    return set()


class Multiply(AbsFlexibleBinaryNode, OperandHelper):

    def __init__(self, op1:IOperator, op2:IOperator):
        super().__init__(op1, op2)
        assert len(dfs_search(op1) & dfs_search(op2)) == 0, "Cannot multiply to elements with same child node."
        # check shape
        self.output_shape()

    def set_input(self, op: IOperator, *op_other):
        raise NotImplementedError("Operation multiply doesn't support this operation.")

    def output_shape(self) -> [list, tuple, None]:
        if self.op_left.output_shape() is None or self.op_right.output_shape() is None:
            return None

        if len(self.op_left.output_shape()) != 2 or len(self.op_right.output_shape()) != 2\
            or self.op_left.output_shape()[1] != self.op_right.output_shape()[0]:
            raise AssertionError("Operation cannot brand between {} and {}".format(self.op_left.output_shape(), self.op_right.output_shape()))

        return [self.op_left.output_shape()[0], self.op_right.output_shape()[1]]

    def do_forward(self, left:[float, ndarray], right:[float, ndarray], training:bool=True) -> [float, ndarray]:
        return np.dot(left, right)

    def do_backward(self, left:[float, ndarray], right:[float, ndarray], grad:[float, ndarray]) -> [ndarray, float]:
        return np.dot(grad, right.T), np.dot(left.T, grad)
