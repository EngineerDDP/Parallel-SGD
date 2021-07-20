import unittest

import numpy as np

import nn
from nn.layer.maxpool import MaxPool


class MyTestCase(unittest.TestCase):

    def test_normal(self):
        x = nn.value.Variable(shape=(2, 6, 6, 1))
        y = np.zeros(shape=(2, 3, 3, 1))

        max_pool = MaxPool(strides=(2, 2), padding="VALID", size=(2, 2), inputs=x)

        res = max_pool.F()
        self.assertEqual(res.shape, (2, 3, 3, 1))

        max_pool.G(y)
        self.assertEqual(x.get_gradient().shape, x.get_shape())

        self.assertEqual(max_pool.output_shape(), (2, 3, 3, 1))
        self.assertEqual(max_pool.variables, ())
        print(max_pool.__repr__())
        print(max_pool.__str__())

    def test_padding(self):
        x = nn.value.Variable(shape=(2, 6, 6, 1))
        y = MaxPool(strides=(2, 2), padding="SAME", size=(2, 2), inputs=x)
        self.assertEqual(y.F().shape, (2, 3, 3, 1))

    def test_strides(self):
        x = nn.value.Variable(shape=(2, 6, 6, 1))
        y = MaxPool(strides=(3, 3), padding="SAME", size=(2, 2), inputs=x)

        self.assertEqual(y.F().shape, (2, 2, 2, 1))

if __name__ == '__main__':
    unittest.main()
