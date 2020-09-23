from numpy import ndarray
from nn.abstract import AbsFlexibleBinaryNode
import tensorflow as tf

class Conv2D(AbsFlexibleBinaryNode):
    def __init__(self, strides, padding):
        super().__init__()
        self.__grad = 0
        self.__strides = strides
        self.__padding = padding
        self.__out_shape = 0

    def do_forward(self, left: [float, ndarray], right: [float, ndarray], training: bool = True) -> [float, ndarray]:
        left = tf.constant(left, dtype=tf.float32)
        right = tf.Variable(tf.constant(right, dtype=tf.float32))
        with tf.GradientTape() as tape:
            out = tf.nn.conv2d(left, right, self.__strides, self.__padding)
        self.__grad = tape.gradient(out, right)
        self.__out_shape = out.numpy().shape
        return out

    def do_backward(self, left: [float, ndarray], right: [float, ndarray], grad: [float, ndarray]) -> (ndarray, ndarray):

        return self.__grad.numpy() * grad

    def output_shape(self) -> [list, tuple, None]:

        return self.__out_shape