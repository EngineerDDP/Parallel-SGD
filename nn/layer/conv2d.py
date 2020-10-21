from typing import Sequence, Union, Optional

import numpy as np
import tensorflow as tf

from nn.activation.interface import IActivation
from nn.interface import IOperator
from nn.layer.abstract import AbsLayer
from nn.value.trainable import Weights


class Conv2D(AbsLayer):
    """
        Convolve 2D layer.
        Base operator implemented by tensorflow.
    """

    def __init__(self, kernel: int, kernel_size: Sequence[int], strides: Optional[Sequence[int]] = None,
                 activation: IActivation = None, inputs: IOperator = None):
        """
            Currently support "VALID" convolve only.
        :param kernel: Kernel count
        :param kernel_size: kernel size, [height, width]
        :param strides: strikes of convolve operation, [height, width]
        :param activation: activation function, None indicates that this layer use linear activation.
        :param inputs: input operator. IOperator instance.
        """
        super().__init__(inputs, activation)
        if strides is None:
            strides = [1, 1]
        self.__kernel = Weights()
        self.__bias = Weights()
        self.__count_kernel: int = kernel
        self.__size_kernel: Sequence[int] = kernel_size
        self.__shape_kernel: Sequence[int] = ()
        self.__strides: Sequence[int] = strides
        self.__padding: Union[Sequence[int], str] = "VALID"
        self.__shape_output: [Sequence[int]] = None
        self.__padding_kernel: [Sequence[int]] = None
        if inputs and inputs.output_shape():
            self.__get_shape(inputs.output_shape())

    @property
    def variables(self) -> tuple:
        return self.__kernel, self.__bias

    def __get_shape(self, input_shape: Sequence[int]):
        s_h = self.__strides[0]
        s_w = self.__strides[1]
        k_h = self.__size_kernel[0]
        k_w = self.__size_kernel[1]
        x_h = input_shape[1]
        x_w = input_shape[2]
        out_h = 1 + (x_h - k_h) // s_h
        out_w = 1 + (x_w - k_w) // s_w
        pad_h = ((x_h - 1) * s_h + out_h) // 2
        pad_w = ((x_w - 1) * s_w + out_w) // 2
        self.__shape_output = (-1, out_h, out_w, self.__count_kernel)
        self.__padding_kernel = [pad_h, pad_w]
        self.__shape_kernel = (*self.__size_kernel, input_shape[3], self.__count_kernel)

    def initialize_parameters(self, x: np.ndarray) -> None:
        # update current shape
        self.__get_shape(x.shape)
        nk = self.__size_kernel[0] * self.__size_kernel[1] * self.__shape_kernel[2]
        nk_1 = self.__size_kernel[0] * self.__size_kernel[1] * self.__count_kernel

        high = np.sqrt(6 / (nk + nk_1))
        low = -high
        self.__kernel.set_value(np.random.uniform(low=low, high=high, size=self.__shape_kernel))
        self.__bias.set_value(np.zeros(shape=self.__shape_output[1:]))

    def do_forward_predict(self, x: np.ndarray):
        tf_input = tf.Variable(tf.constant(x, dtype=tf.float32))
        tf_kernel = tf.Variable(tf.constant(self.__kernel.get_value(), dtype=tf.float32))
        tf_out = tf.nn.conv2d(tf_input, tf_kernel, self.__strides, self.__padding)
        out = tf_out.numpy()
        return out.astype('float64') + self.__bias.get_value()

    def do_forward_train(self, x):
        return self.do_forward_predict(x)

    def backward_adjust(self, grad) -> None:
        tf_input = tf.constant(self.input_ref, dtype=tf.float32)
        tf_grad = tf.constant(grad, dtype=tf.float32)
        tf_out = tf.nn.conv2d(tf.transpose(tf_input, perm=[3, 1, 2, 0]),
                              tf.transpose(tf_grad, perm=[1, 2, 0, 3]), self.__strides, "VALID")
        out = tf.transpose(tf_out, perm=[1, 2, 0, 3]).numpy()
        self.__kernel.adjust(out)
        self.__bias.adjust(grad)

    def backward_propagate(self, grad):
        tf_kernel = tf.constant(self.__kernel.get_value(), dtype=tf.float32)
        tf_grad = tf.constant(grad, dtype=tf.float32)
        tf_out = tf.nn.conv2d_transpose(tf_grad, tf_kernel, self.input_ref.shape, self.__strides, self.__padding)
        grad = tf_out.numpy()
        return grad.astype('float64')

    def output_shape(self) -> [list, tuple, None]:
        return self.__shape_output

    def __str__(self):
        return "<2D Convolution Layer, kernel size: {}, count: {}>".format(self.__size_kernel, self.__count_kernel)

    def __repr__(self):
        return self.__str__()
