import numpy as np

from nn.interface import IOperator
from nn.loss.abstract import AbsLoss


class MSELoss(AbsLoss):

    def __init__(self, op1: IOperator, op2: IOperator):
        super().__init__(op1, op2)

    def do_backward(self, arg1, arg2):
        return 2.0 * (arg1 - arg2), -2.0 * (arg1 - arg2)

    def do_forward(self, arg1, arg2):
        return np.sum(np.square(arg1 - arg2))
