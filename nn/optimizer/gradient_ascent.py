from nn.interface import IOptimizer, IValue
from nn.optimizer import IGradientDescent


class GAOptimizer(IOptimizer):

    def __init__(self, gd_optimizer: IGradientDescent):
        self.__optimizer = gd_optimizer
        self.__batch_size = 1

    def optimize(self, variable: IValue) -> None:
        """
            1st order gradient based optimize algorithm.
            {arg max}_{x}{f(x)}
        :param variable: variable object.
        :return: None
        """
        variable.set_value(variable.get_value() + self.__optimizer.delta(variable.get_gradient() / self.__batch_size))

    def set_batch_size(self, batch_size:int):
        self.__batch_size = batch_size

    def __str__(self):
        return "<GDOptimizer, Using {}>".format(self.__optimizer)
