from nn.interface import IOptimizer, ITrainable
from nn.gradient_descent.interface import IGradientDescent


class GDOptimizer(IOptimizer):

    def __init__(self, gd_optimizer: IGradientDescent):
        self.__optimizer = gd_optimizer
        self.__batch_size = 1

    def optimize(self, variable: ITrainable) -> None:
        """
            1st order gradient based optimize algorithm.
            {arg min}_{x}{F(x)}
        :param variable: variable object.
        :return: None
        """
        grad = variable.get_gradient()
        if variable.get_shape() != grad.shape:
            grad = grad.sum(axis=0)
        variable.set_value(variable.get_value() - self.__optimizer.delta(grad / self.__batch_size))

    def set_batch_size(self, batch_size: int):
        self.__batch_size = batch_size

    def __str__(self):
        return "<GDOptimizer, Using {}>".format(self.__optimizer)
