import time
import unittest

import numpy as np
from parallel_sgd.codec.utils import quant


class MyTestCase(unittest.TestCase):

    def test_cpp_code(self):
        arr = [-3, -2, -1, 0, 1.5, 2, 3]

        q = quant(arr)
        rnd = np.random.uniform(low=0.5, high=0.5, size=[1000000])

        s = time.time()
        for i in range(100):
            b = q.stochastic(rnd)
            a = q.decode(b)
            # 量化误差限制，不能高过百分之一
            self.assertLess(np.mean(rnd - a), 0.005)
        e = time.time() - s

        # 量化时间限制，不能太慢
        self.assertLess(e, 3.5)


if __name__ == '__main__':
    unittest.main()
