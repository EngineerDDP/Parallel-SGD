import numpy as np


for f in range(10000):
    for n in range(10000):
        if f >= n:
            try:
                left = np.math.factorial(f/2) / (np.math.factorial(f*3/(2*n)) * np.math.factorial(f/2 - (f*3/(2*n))))
                right = n
                if left == right:
                    print('f = ', f, 'n = ', n)
            except:
                continue




