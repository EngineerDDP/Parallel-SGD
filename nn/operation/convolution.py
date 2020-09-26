import tensorflow as tf
import numpy as np

from typing import List, Tuple
from numpy import ndarray
from nn.abstract import AbsFlexibleBinaryNode
from nn.interface import IOperator


class Conv2D(AbsFlexibleBinaryNode):

    def __init__(self, op1: IOperator, op2: IOperator, strides: [List[int], Tuple[int]], padding: [List[int], Tuple[int], str]):
        super().__init__(op1, op2)
        self.__grad_left: [ndarray] = None
        self.__grad_right: [ndarray] = None
        self.__strides: [List[int], Tuple[int]] = strides
        self.__padding: [List[int], Tuple[int], str] = padding
        self.__out_shape: [List[int], Tuple[int]] = None

    def do_forward(self, left: [float, ndarray], right: [float, ndarray], training: bool = True) -> [float, ndarray]:
        left = tf.Variable(tf.constant(left, dtype=tf.float32))
        right = tf.Variable(tf.constant(right, dtype=tf.float32))
        with tf.GradientTape() as tape:
            out = tf.nn.conv2d(left, right, self.__strides, self.__padding)
        self.__grad_left, self.__grad_right = tape.gradient(out, (left, right))
        self.__out_shape = out.numpy().shape
        return out.numpy()

    def do_backward(self, left: [float, ndarray], right: [float, ndarray], grad: [float, ndarray]) -> (ndarray, ndarray):
        return np.multiply(self.__grad_left.numpy(), grad), np.multiply(self.__grad_right.numpy(), grad)

    def output_shape(self) -> [list, tuple, None]:
        return self.__out_shape


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from nn.value import Variable
    x = Variable(shape=(1,5,5,1))
    w = Variable(shape=(2,2,1,2))
    y = Conv2D(x, w ,[1,1,1,1],"VALID")
    print(y.F())
    # x_in = np.array([[
    #     [[2], [1], [2], [0], [1]],
    #     [[1], [3], [2], [2], [3]],
    #     [[1], [1], [3], [3], [0]],
    #     [[2], [2], [0], [1], [1]],
    #     [[0], [0], [3], [1], [2]], ]])
    # kernel_in = np.array([
    #     [[[2, 0.1]], [[3, 0.2]]],
    #     [[[0, 0.3]], [[1, 0.4]]], ])
    # x = tf.constant(x_in, dtype=tf.float32)
    # kernel = tf.constant(kernel_in, dtype=tf.float32)
    # print(tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID'))

