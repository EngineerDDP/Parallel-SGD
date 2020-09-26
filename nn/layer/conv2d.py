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
        self.__grad_left = None
        self.__grad_right = None
        self.__out_shape = None
        self.__tape: tf.GradientTape = None
        self.__in = None

    @property
    def variables(self) -> tuple:
        return self.__kernal,

    def initialize_parameters(self, x) -> None:
        if self.__kernal.get_value() is None:
            self.__kernal.set_value(np.random.uniform(low=-1, high=1, size=self.__size))

    def do_forward_predict(self, x):
        self.__in = x
        left = tf.Variable(tf.constant(x, dtype=tf.float32))
        right = tf.Variable(tf.constant(self.__kernal.get_value(), dtype=tf.float32))
        out = tf.nn.conv2d(left, right, self.__strides, self.__padding)
        self.__out_shape = out.numpy().shape
        return out.numpy()

    def do_forward_train(self, x):
        return self.do_forward_predict(x)

    def backward_adjust(self, grad) -> None:
        left = tf.constant(self.__in, dtype=tf.float32)
        grad = tf.constant(grad, dtype=tf.float32)
        out = tf.nn.conv2d(tf.transpose(left[:,::-1,::-1,:], perm=[3,1,2,0]), tf.transpose(grad, perm=[1,2,0,3]), self.__strides, self.__padding)
        out = tf.transpose(out, perm=[1,2,0,3]).numpy()
        self.__kernal.adjust(out)

    def backward_propagate(self, grad):
        left = tf.constant(self.__kernal.get_value(), dtype=tf.float32)
        grad = tf.constant(grad, dtype=tf.float32)
        out = tf.nn.conv2d_transpose(grad, left, self.__in.shape, self.__strides, self.__padding)
        out = out.numpy()
        return out
        # return self.__grad_left.numpy()

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
    y = Conv2DLayer([1, 1, 1, 1], "VALID", (2, 2, 1, 2), inputs=x)

    print(y.F(state=ModelState.Training).shape)
    y.G(np.ones((1,4,4,2)))
