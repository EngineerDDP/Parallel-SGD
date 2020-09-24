import numpy as np

from typing import List, Tuple
from nn.interface import IOperator
from nn.layer.abstract import Weights, AbsLayer
from nn.activation.interface import IActivation


class MaxPool(AbsLayer):
    def __init__(self, filter_size: [List[int], Tuple[int]], strikes: [List[int], Tuple[int]], activation:IActivation=None, input:IOperator=None):
        super().__init__(input, activation)
        self.__strikes: [List[int], Tuple[int]] = strikes
        self.__filter_size: [List[int], Tuple[int]] = filter_size
        self.__shape_out: [List[int], Tuple[int]] = None
        self.__x: [List[int], Tuple[int]] = None

    @property
    def variables(self) -> tuple:
        return ()

    def initialize_parameters(self, x) -> None:
        pass

    def do_forward_predict(self, x):
        # build buffer for result
        self.__x = x
        result_height = x.shape[0] // self.__strikes[0]
        result_width = x.shape[1] // self.__strikes[1]
        self.__shape_out = (result_height, result_width)
        result = np.zeros(shape=[result_height, result_width])
        # calculate result
        for i in range(result_height):
            for j in range(result_width):
                # pooling window up
                row_start = i * self.__strikes[0]
                # pooling window bottom
                row_end = row_start + self.__filter_size[0]
                # window left
                col_start = j * self.__strikes[1]
                # window right
                col_end = col_start + self.__filter_size[1]
                # pooling with specified action
                result[i, j] = np.max(x[row_start:row_end, col_start:col_end])

        return result

    def do_forward_train(self, x):
        return self.do_forward_predict(x)

    def backward_adjust(self, grad) -> None:
        pass

    def backward_propagate(self, grad):
        # build result
        result = np.zeros_like(self.__x)
        # do up-sampling
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                # pooling window up
                row_start = i * self.__strikes[0]
                # pooling window bottom
                row_end = row_start + self.__filter_size[0]
                # window left
                col_start = j * self.__strikes[1]
                # window right
                col_end = col_start + self.__filter_size[1]
                # get arg max
                idx = np.max(self.__x[row_start:row_end, col_start:col_end])
                # save
                result[row_start:row_end, col_start:col_end][idx] = grad[i, j]

        return result

    def output_shape(self) -> [list, tuple, None]:
        return self.__shape_out

    def __str__(self):
        return "<MaxPool Layer, filter_size: {}>".format(self.__filter_size)

    def __repr__(self):
        print(self.__str__())
