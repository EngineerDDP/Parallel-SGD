import numpy as np
import tensorflow as tf

from typing import List, Tuple, Sequence, Union, Iterable, Optional
from nn.interface import IOperator
from nn.layer.abstract import Weights, AbsLayer
from nn.activation.interface import IActivation


class Conv2DLayer(AbsLayer):

    def __init__(self, kernel: int, kernel_size: Sequence[int], strides: Optional[Sequence[int]] = None,
                 padding: Union[Sequence[int], str] = "VALID", activation: IActivation = None,
                 inputs: IOperator = None):
        super().__init__(inputs, activation)
        if strides is None:
            strides = [1, 1, 1, 1]
        self.__kernel = Weights()
        self.__bias = Weights()
        self.__count_kernel: int = kernel
        self.__size_kernel: Sequence[int] = kernel_size
        self.__shape_kernel: Sequence[int] = ()
        self.__strides: Sequence[int] = strides
        self.__padding: Union[Sequence[int], str] = padding
        self.__p_h = None
        self.__p_w = None
        self.__shape_output: [Sequence[int]] = None
        if inputs and inputs.output_shape():
            self.__get_shape(inputs.output_shape())

    @property
    def variables(self) -> tuple:
        return self.__kernel, self.__bias

    def __get_shape(self, input_shape: Sequence[int]):
        s_h = self.__strides[1]
        s_w = self.__strides[2]
        k_h = self.__size_kernel[0]
        k_w = self.__size_kernel[1]
        x_h = input_shape[1]
        x_w = input_shape[2]
        out_h = (x_h - k_h + 1) // s_h
        out_w = (x_w - k_w + 1) // s_w
        self.__shape_output = (input_shape[0], out_h, out_w, self.__count_kernel)
        if self.__padding == "SAME":
            self.__p_h = (s_h * x_h + k_h - 1 - x_h) // (2 * s_h)
            self.__p_w = (s_w * x_w + k_w - 1 - x_w) // (2 * s_w)
            # self.__padding = [[0, 0], [0, 0], [p_h, p_h], [p_w, p_w]]
        self.__shape_kernel = (*self.__size_kernel, input_shape[3], self.__count_kernel)

    def initialize_parameters(self, x: np.ndarray) -> None:
        # update current shape
        self.__get_shape(x.shape)
        nk, nk_1 = 1, 1
        for i, j in zip(self.__shape_kernel[:3], self.__shape_output[-3:]):
            nk, nk_1 = nk * i, nk_1 * j

        high = np.sqrt(6 / (nk + nk_1))
        low = -high
        self.__kernel.set_value(np.random.uniform(low=low, high=high, size=self.__shape_kernel))
        self.__bias.set_value(np.zeros(shape=self.__shape_output[1:]))

    def do_forward_predict(self, x: np.ndarray):
        left = tf.Variable(tf.constant(x, dtype=tf.float32))
        right = tf.Variable(tf.constant(self.__kernel.get_value(), dtype=tf.float32))
        out = tf.nn.conv2d(left, right, self.__strides, self.__padding)
        # self.__out_shape = out.numpy().shape
        return out.numpy() + self.__bias.get_value()

    def do_forward_train(self, x):
        return self.do_forward_predict(x)

    def backward_adjust(self, grad) -> None:
        tf_input = tf.constant(self.input_ref, dtype=tf.float32)
        if self.__padding == "SAME":
            tf_input = tf.pad(tf_input, [[0, 0], [self.__p_h, self.__p_h], [self.__p_w, self.__p_w], [0, 0]])
        tf_grad = tf.constant(grad, dtype=tf.float32)
        tf_out = tf.nn.conv2d(tf.transpose(tf_input[:, ::-1, ::-1, :], perm=[3, 1, 2, 0]),
                              tf.transpose(tf_grad, perm=[1, 2, 0, 3]), self.__strides, "VALID")
        out = tf.transpose(tf_out, perm=[1, 2, 0, 3]).numpy()
        self.__kernel.adjust(out / np.sum(self.output_shape()[1:3]))
        self.__bias.adjust(grad)

    def backward_propagate(self, grad):
        tf_kernel = tf.constant(self.__kernel.get_value(), dtype=tf.float32)
        tf_grad = tf.constant(grad, dtype=tf.float32)
        tf_out = tf.nn.conv2d_transpose(tf_grad, tf_kernel, self.input_ref.shape, self.__strides, self.__padding)
        grad = tf_out.numpy()
        return grad

    def output_shape(self) -> [list, tuple, None]:
        return self.__shape_output

    def __str__(self):
        return "<2D Convolution Layer, kernel size: {}, count: {}>".format(self.__size_kernel, self.__count_kernel)

    def __repr__(self):
        return self.__str__()
