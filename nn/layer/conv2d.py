import numpy as np
import tensorflow as tf

from typing import List, Tuple, Iterable, Union
from nn.interface import IOperator
from nn.layer.abstract import Weights, AbsLayer
from nn.activation.interface import IActivation


class Conv2DLayer(AbsLayer):

    def __init__(self, strides: Iterable[int], padding: Union[Iterable[int], str],
                 size: Iterable[int], activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__kernal = Weights()
        self.__strides: [List[int], Tuple[int]] = strides
        self.__padding: [List[int], Tuple[int], str] = padding
        self.__size: [List[int], Tuple[int]] = size
        self.__out_shape = None
        self.__in_shape = None
        self.__input = None
        self.__p_h = None
        self.__p_w = None

    @property
    def variables(self) -> tuple:
        return self.__kernal,

    def initialize_parameters(self, x) -> None:
        self.__kernal.set_value(np.random.uniform(low=-1, high=1, size=self.__size))
        s_h = self.__strides[1]
        s_w = self.__strides[2]
        k_h = self.__size[0]
        k_w = self.__size[1]
        x_h = x.shape[1]
        x_w = x.shape[2]
        out_h = (x_h - k_h + 1) / s_h
        out_w = (x_w - k_w + 1) / s_w
        self.__in_shape = x.shape
        self.__out_shape = (x.shape[0], out_h, out_w, self.__size[-1])
        if self.__padding == "SAME":
            self.__p_h = (s_h * x_h + k_h - 1 - x_h) // (2 * s_h)
            self.__p_w = (s_w * x_w + k_w - 1 - x_w) // (2 * s_w)
            # self.__padding = [[0, 0], [0, 0], [p_h, p_h], [p_w, p_w]]

    def do_forward_predict(self, x):
        self.__input = x
        left = tf.Variable(tf.constant(x, dtype=tf.float32))
        right = tf.Variable(tf.constant(self.__kernal.get_value(), dtype=tf.float32))
        out = tf.nn.conv2d(left, right, self.__strides, self.__padding)
        # self.__out_shape = out.numpy().shape
        return out.numpy()

    def do_forward_train(self, x):
        return self.do_forward_predict(x)

    def backward_adjust(self, grad) -> None:
        left = tf.constant(self.__input, dtype=tf.float32)
        if self.__padding == "SAME":
            left = tf.pad(left,[[0, 0], [self.__p_h, self.__p_h], [self.__p_w, self.__p_w], [0, 0]])
        grad = tf.constant(grad, dtype=tf.float32)
        out = tf.nn.conv2d(tf.transpose(left[:,::-1,::-1,:], perm=[3,1,2,0]), tf.transpose(grad, perm=[1,2,0,3]), self.__strides, "VALID")
        out = tf.transpose(out, perm=[1,2,0,3]).numpy()
        self.__kernal.adjust(out)

    def backward_propagate(self, grad):
        left = tf.constant(self.__kernal.get_value(), dtype=tf.float32)
        grad = tf.constant(grad, dtype=tf.float32)
        out = tf.nn.conv2d_transpose(grad, left, self.__input.shape, self.__strides, self.__padding)
        out = out.numpy()
        return out


    def output_shape(self) -> [list, tuple, None]:
        return self.__out_shape

    def __str__(self):
        return "<Conv2D Layer, kernel: {}>".format(self.__size)

    def __repr__(self):
        print(self.__str__())


if __name__ == '__main__':
    import os
    from nn.interface import ModelState
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from nn.value import Variable

    x = Variable(shape=(1, 5, 5, 1))
    y = Conv2DLayer([1, 1, 1, 1], "SAME", (3, 3, 1, 2), inputs=x)

    print(y.F(state=ModelState.Training).shape)
    y.G(np.ones((1,5,5,2)))
