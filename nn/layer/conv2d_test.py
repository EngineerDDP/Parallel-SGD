import unittest

import numpy as np

from nn.layer.conv2d import Conv2D


class TestConv(unittest.TestCase):

    def test_conv(self):
        x = np.ones(shape=[100, 32, 32, 3])
        conv_layer = Conv2D(64, [3, 3])
        for _ in range(100):
            conv_layer.F(x)
            y = np.zeros(shape=[100, 30, 30, 64])
            conv_layer.G(y)
