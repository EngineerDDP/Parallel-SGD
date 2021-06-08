from typing import Sequence

import tensorflow as tf

from nn.activation.interface import IActivation
from nn.interface import IOperator
from nn.layer.abstract import AbsLayer


class MaxPool(AbsLayer):

    def __init__(self, strides: Sequence[int], padding: [Sequence[int], str],
                 size: Sequence[int], activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__strides: Sequence = strides
        self.__padding: [Sequence, str] = padding
        self.__size: Sequence = size
        self.__mask = None
        self.__out_shape = None
        self.__in_shape = None

    @property
    def variables(self) -> tuple:
        return ()

    def initialize_parameters(self, x) -> None:
        pass

    def do_forward_predict(self, x):
        left = tf.Variable(tf.constant(x, dtype=tf.float32))
        with tf.GradientTape() as tape:
            out = tf.nn.max_pool2d(left, self.__size, self.__strides, self.__padding)
        self.__mask = tape.gradient(out, left)
        self.__out_shape = out.numpy().shape
        self.__in_shape = x.shape
        return out.numpy()

    def do_forward_train(self, x):
        return self.do_forward_predict(x)

    def backward_adjust(self, grad) -> None:
        pass

    def backward_propagate(self, grad):
        indices = tf.where(self.__mask > 0)
        updates = tf.reshape(tf.constant(grad), (-1))
        shape = tf.constant(self.__in_shape, dtype=tf.int64)
        return tf.scatter_nd(indices, updates, shape).numpy()

    def output_shape(self) -> [list, tuple, None]:
        return self.__out_shape

    def __str__(self):
        return "<MaxPool Layer, filter_size: {}>".format(self.__size)

    def __repr__(self):
        return "<MaxPool Layer, filter_size: {}>".format(self.__size)
