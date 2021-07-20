import unittest

import numpy as np

import nn
from nn.layer.conv2d import Conv2D


class TestConv(unittest.TestCase):

    def test_normal(self):
        x = nn.value.Placeholder(shape=[1, 32, 32, 3])
        x.set_value(np.ones(shape=[1, 32, 32, 3]))
        y = np.zeros(shape=[1, 30, 30, 64])

        conv_layer = Conv2D(64, [3, 3])
        conv_layer.set_input(x)

        res = conv_layer.F(x)
        self.assertEqual(res.shape, (1, 30, 30, 64))

        conv_layer.G(y)
        self.assertEqual(x.get_gradient().shape, x.get_shape())

    def test_strides(self):
        x = nn.value.Placeholder(shape=[1, 32, 32, 3])
        x.set_value(np.ones(shape=[1, 32, 32, 3]))

        conv_layer = Conv2D(1, [5, 5], strides=(1, 1))
        y = np.zeros(shape=[1, 28, 28, 1])
        conv_layer.set_input(x)

        conv_layer.F()
        conv_layer.G(y)

        self.assertEqual(x.get_gradient().shape, x.get_shape())

        conv_layer = Conv2D(1, [5, 5], strides=(2, 2))
        y = np.zeros(shape=[1, 14, 14, 1])
        conv_layer.set_input(x)

        conv_layer.F()
        conv_layer.G(y)

        self.assertEqual(x.get_gradient().shape, x.get_shape())

    def test_padding(self):
        x = nn.value.Placeholder(shape=[1, 32, 32, 3])
        x.set_value(np.ones(shape=[1, 32, 32, 3]))

        conv_layer = Conv2D(1, [5, 5], padding='SAME')
        y = np.zeros(shape=[1, 32, 32, 1])
        conv_layer.set_input(x)

        conv_layer.F()
        conv_layer.G(y)

        self.assertEqual(x.get_gradient().shape, x.get_shape())

        conv_layer = Conv2D(1, [5, 5], padding='VALID')
        y = np.zeros(shape=[1, 28, 28, 1])
        conv_layer.set_input(x)

        conv_layer.F()
        conv_layer.G(y)

        self.assertEqual(x.get_gradient().shape, x.get_shape())
