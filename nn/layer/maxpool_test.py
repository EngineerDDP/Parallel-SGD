import unittest

import nn
from nn.layer.maxpool import MaxPool


class MyTestCase(unittest.TestCase):
    def test_normal(self):
        x = nn.value.Variable(shape=(2, 5, 5, 1))
        y = MaxPool([1, 1, 1, 1], "VALID", (2, 2), inputs=x)
        self.assertEqual(y.F().shape, (2, 3, 3, 1))


if __name__ == '__main__':
    unittest.main()
