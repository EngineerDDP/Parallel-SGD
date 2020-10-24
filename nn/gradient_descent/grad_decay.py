import numpy as np

from nn.gradient_descent.interface import IGradientDescent


class GradientDecay(IGradientDescent):

    def __init__(self, learning_rate: float = 0.03):
        self.__learn_rate: float = learning_rate
        self.__t: int = 0

    def delta(self, var: np.ndarray) -> np.ndarray:
        self.__t += 1
        return self.__learn_rate * var / np.sqrt(self.__t)
