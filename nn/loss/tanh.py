import numpy as np

from nn.interface import IOperator
from nn.loss.abstract import AbsLoss


class TanhLoss(AbsLoss):

    def __init__(self, op1: IOperator, op2: IOperator):
        super().__init__(op1, op2)

    def do_forward(self, arg1, arg2):
        return np.mean(np.square(np.tanh(arg2 - arg1)))

    def do_backward(self, arg1, arg2):
        tmp_tanh = np.tanh(arg2 - arg1)
        return -2.0 * np.multiply(tmp_tanh, (1 - np.square(tmp_tanh))), 2.0 * np.multiply(tmp_tanh, (1 + np.square(tmp_tanh)))
