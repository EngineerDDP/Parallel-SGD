import numpy as np

from typing import List, Tuple
from nn.interface import IOperator
from nn.layer.abstract import Weights, AbsLayer
from nn.activation.interface import IActivation


class Reshape(AbsLayer):

    def __init__(self, shape: [List[int], Tuple[int]], activation: IActivation = None, input: IOperator = None):
        super().__init__(input, activation)
        self.__shape_out: [List[int], Tuple[int]] = shape
        self.__shape_in: [List[int], Tuple[int]] = input.output_shape()

    @property
    def variables(self) -> tuple:
        return ()

    def initialize_parameters(self, x) -> None:
        pass

    def do_forward_predict(self, x):
        return np.reshape(x, self.__shape_out)

    def do_forward_train(self, x):
        return np.reshape(x, self.__shape_out)

    def backward_adjust(self, grad) -> None:
        pass

    def backward_propagate(self, grad):
        return np.reshape(grad, self.__shape_in)

    def output_shape(self) -> [list, tuple, None]:
        return self.__shape_out

    def __str__(self):
        return "<ReShape Layer, shape: {}>".format(self.__shape_out)

    def __repr__(self):
        print(self.__str__())
