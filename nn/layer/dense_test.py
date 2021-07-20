import unittest

import numpy as np

import nn
from nn.layer.dense import Dense


class TestDense(unittest.TestCase):

    def test_normal(self):
        x = nn.value.Placeholder(shape=[100, 64])
        x.set_value(np.ones(shape=[100, 64]))
        y = np.zeros(shape=[100, 10])

        dense_layer = Dense(units=10)
        dense_layer.set_input(x)

        res = dense_layer.F(x)
        self.assertEqual(res.shape, y.shape)

        dense_layer.G(y)
        self.assertEqual(x.get_gradient().shape, x.get_shape())
