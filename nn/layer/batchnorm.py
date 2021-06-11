import numpy as np

from nn.activation.interface import IActivation
from nn.interface import IOperator
from nn.layer.abstract import AbsLayer


class BatchNorm(AbsLayer):

    def __init__(self, activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__shape = None
        self.__mu = 0
        self.__sigma = 1
        self.__eps = 1e-7
        self.__momentum = 0.7

    def output_shape(self) -> [list, tuple, None]:
        return self.__shape

    @property
    def variables(self) -> tuple:
        return ()

    def initialize_parameters(self, x) -> None:
        self.__shape = x.shape

    def do_forward_predict(self, x):
        return (x - self.__mu) / np.sqrt(self.__sigma + self.__eps)

    def do_forward_train(self, x):
        self.__mu = self.__momentum * self.__mu + (1 - self.__momentum) * x.mean(axis=0)
        self.__sigma = self.__momentum * self.__sigma + (1 - self.__momentum) * x.var(axis=0)
        return self.do_forward_predict(x)

    def backward_adjust(self, grad) -> None:
        pass

    def backward_propagate(self, grad):
        return grad / np.sqrt(self.__sigma + self.__eps)

    def __str__(self):
        return "<BatchNorm Layer>"

    def __repr__(self):
        return "<BatchNorm Layer>"
