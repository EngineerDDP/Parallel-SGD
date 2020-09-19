import numpy as np

from nn.interface import IOperator
from nn.loss.abstract import AbsLoss


class Cross_Entropy(AbsLoss):

    def __init__(self, op1: IOperator, op2: IOperator):
        super().__init__(op1, op2)

    def do_forward(self, arg1, arg2):
        return np.mean(np.multiply(arg1, np.log(arg2)) + np.multiply((1 - arg1), np.log(1 - arg2))) * -1

    def do_backward(self, arg1, arg2):
        return (1 - arg2) / (1 - arg1) - arg2 / arg1, (1 - arg1) / (1 - arg2) - arg1 / arg2


class Cross_Entropy_With_Softmax(AbsLoss):

    def __init__(self, op1: IOperator, op2: IOperator):
        super().__init__(op1, op2)

    def do_forward(self, arg1, arg2):
        return np.mean(np.multiply(arg1, np.log(arg2)) + np.multiply((1 - arg1), np.log(1 - arg2))) * -1

    def do_backward(self, arg1, arg2):
        return arg1 - arg2, arg2 - arg1
