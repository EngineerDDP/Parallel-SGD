import unittest

import numpy as np

import nn
from nn.layer.conv2d import Conv2D


class TestConv(unittest.TestCase):

    def test_conv(self):
        x = np.ones(shape=[100, 32, 32, 3])
        conv_layer = Conv2D(64, [3, 3])
        for _ in range(100):
            conv_layer.F(x)
            y = np.zeros(shape=[100, 30, 30, 64])
            conv_layer.G(y)

    def test_strides(self):
        x = nn.value.Placeholder(shape=[1, 32, 32, 3])
        x.set_value(np.ones(shape=[1, 32, 32, 3]))
        y = np.zeros(shape=[1, 28, 28, 1])
        conv_layer = Conv2D(1, [5, 5], strides=(1, 1))
        conv_layer.set_input(x)

        conv_layer.F()
        conv_layer.G(y)

        self.assertEqual(x.get_gradient().shape, x.get_shape())

        conv_layer = Conv2D(1, [5, 5], strides=(2, 2))
        conv_layer.set_input(x)

        conv_layer.F()
        conv_layer.G(y)

        self.assertEqual(x.get_gradient().shape, x.get_shape())
