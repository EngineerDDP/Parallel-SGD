import numpy as np

from nn.interface import IOperator
from nn.layer.abstract import Weights, AbsLayer
from nn.activation.interface import IActivation


class BatchNorm(AbsLayer):

    def __init__(self, activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__shape = None
        self.__mu = None
        self.__sigma = None
        self.__eps = 1e-7

    def output_shape(self) -> [list, tuple, None]:
        return self.__shape

    @property
    def variables(self) -> tuple:
        return self.__mu, self.__sigma

    def initialize_parameters(self, x) -> None:
        self.__shape = x.shape

    def do_forward_predict(self, x):
        self.__mu = x.mean(axis=0)
        self.__sigma = x.var(axis=0)
        return (x - self.__mu) / np.sqrt(self.__sigma + self.__eps)

    def do_forward_train(self, x):
        return self.do_forward_predict(x)

    def backward_adjust(self, grad) -> None:
        pass

    def backward_propagate(self, grad):
        return grad * np.sqrt(self.__sigma + self.__eps) + self.__mu

    def __str__(self):
        return "<BatchNorm Layer>"

    def __repr__(self):
        print(self.__str__())