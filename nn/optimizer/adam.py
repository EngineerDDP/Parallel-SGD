import numpy as np

from nn.optimizer.interface import IGradientDescent


class ADAMOptimizer(IGradientDescent):
    """
        ADAM backward_predict propagation.
        Diederik P.Kingma, Jimmy Lei Ba. ADAM: A Method for Stochastic Optimization. ICLR 2015.
    """

    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        # variable to be optimized
        super().__init__()
        # check the paper for detail
        self.__alpha = alpha
        self.__beta1 = beta_1
        self.__beta2 = beta_2
        self.__epsilon = epsilon
        self.__mt = 0
        self.__vt = 0
        self.__t = 0

    def delta(self, gradient):
        self.__t = self.__t + 1
        self.__mt = self.__beta1 * self.__mt + (1 - self.__beta1) * gradient
        self.__vt = self.__beta2 * self.__vt + (1 - self.__beta2) * np.square(gradient)
        _mt = self.__mt / (1 - self.__beta1 ** self.__t)
        _vt = self.__vt / (1 - self.__beta2 ** self.__t)
        return self.__alpha * _mt / (np.sqrt(_vt) + self.__epsilon)

    def __str__(self):
        return "<Adam Optimizer>"
