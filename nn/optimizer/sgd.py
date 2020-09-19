from nn.optimizer.interface import IGradientDescent


class SGDOptimizer(IGradientDescent):

    def __init__(self, learn_rate=0.01):
        self.__learn_rate = learn_rate

    def delta(self, gradient):
        return gradient * self.__learn_rate

    def __str__(self):
        return "<SGD Optimizer>"
