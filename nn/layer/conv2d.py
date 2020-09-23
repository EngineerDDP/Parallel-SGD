import numpy as np


from typing import List, Tuple
from nn.interface import IOperator
from nn.layer.abstract import Weights, AbsLayer
from nn.activation.interface import IActivation
from nn.operation.convolution import Conv2D as conv


class Conv2DLayer(AbsLayer):

    def __init__(self, strides: [List[int], Tuple[int]], padding: [List[int], Tuple[int], str], size: [List[int], Tuple[int]], activation:IActivation=None, input:IOperator=None):
        super().__init__(input, activation)
        self.__kernal = Weights()
        self.__strides: [List[int], Tuple[int]] = strides
        self.__padding: [List[int], Tuple[int], str] = padding
        self.__size: [List[int], Tuple[int]] = size
        self.__conv: conv = None
    @property
    def variables(self) -> tuple:
        return self.__kernal,

    def initialize_parameters(self, x) -> None:
        if self.__kernal.get_value() is None:
            self.__kernal.set_value(np.random.uniform(low=-1, high=1, size=self.__size))
        self.__conv = conv(self.__strides, self.__padding)

    def do_forward_predict(self, x):
        return self.__conv.do_forward(x, self.__kernal.get_value(), False)

    def do_forward_train(self, x):
        return self.__conv.do_forward(x, self.__kernal.get_value(), True)

    def backward_adjust(self, grad) -> None:
        pass

    def backward_propagate(self, grad):
        return self.__conv.do_backward(None, None, grad)

    def output_shape(self) -> [list, tuple, None]:
        return self.__conv.output_shape()

    def __str__(self):
        return "<Conv2D Layer, Size: {}>".format(self.__size)

    def __repr__(self):
        print(self.__str__())