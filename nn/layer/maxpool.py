import numpy as np
import tensorflow as tf

from typing import List, Tuple
from nn.interface import IOperator
from nn.layer.abstract import Weights, AbsLayer
from nn.activation.interface import IActivation


class MaxPool(AbsLayer):

    def __init__(self, strides: [List[int], Tuple[int]], padding: [List[int], Tuple[int], str],
                 size: [List[int], Tuple[int]], activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__strides: [List[int], Tuple[int]] = strides
        self.__padding: [List[int], Tuple[int], str] = padding
        self.__size: [List[int], Tuple[int]] = size
        self.__grad_left = None
        self.__out_shape = None

    @property
    def variables(self) -> tuple:
        return ()

    def initialize_parameters(self, x) -> None:
        pass

    def do_forward_predict(self, x):
        left = tf.Variable(tf.constant(x, dtype=tf.float32))
        with tf.GradientTape() as tape:
            out = tf.nn.max_pool2d(left, self.__size, self.__strides, self.__padding)
        self.__grad_left = tape.gradient(out, left)
        self.__out_shape = out.numpy().shape
        return out.numpy()

    def do_forward_train(self, x):
        return self.do_forward_predict(x)

    def backward_adjust(self, grad) -> None:
        pass

    def backward_propagate(self, grad):
        return np.multiply(self.__grad_left.numpy(), grad)

    def output_shape(self) -> [list, tuple, None]:
        return self.__out_shape

    def __str__(self):
        return "<MaxPool Layer, filter_size: {}>".format(self.__size)

    def __repr__(self):
        print(self.__str__())


if __name__ == '__main__':
    from nn.value import Variable
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    x = Variable(shape=(2, 5, 5, 1))
    y = MaxPool([1, 1, 1, 1], "VALID", (2, 2), inputs=x)
    print(y.F())
