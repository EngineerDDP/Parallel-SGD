import unittest

import numpy as np

import nn.operation
import nn.value


class TestOP(unittest.TestCase):

    def test_normal(self):
        shape = (2, 2)
        x = nn.value.Variable(np.ones(shape=shape))
        y = nn.value.Variable(np.ones(shape=shape))
        v = nn.value.Placeholder(shape=shape)
        v.set_value(2 * np.ones(shape=shape))

        z = x + y  # 2
        a = z * v  # 4
        b = a ** 3  # 64
        c = b - y  # 63

        r = c.F()
        c.G(np.zeros(shape=shape))

        self.assertEqual(r[0][0], 511)
        self.assertEqual(x.get_gradient().shape, x.get_shape())

    def test_unmatched_multiply(self):
        x = nn.value.Variable(np.ones(shape=(2, 3)))
        y = nn.value.Variable(np.ones(shape=(4, 2)))

        with self.assertRaises(AssertionError):
            x * y

        c = nn.value.Variable(np.ones(shape=(3, 4)))

        z = x * c
        r = z.F()

        self.assertEqual(r.shape, (2, 4))


if __name__ == '__main__':
    unittest.main()
