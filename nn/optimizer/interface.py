from abc import abstractmethod, ABCMeta

from nn.interface import IValue


class IGradientDescent(metaclass=ABCMeta):

    @abstractmethod
    def delta(self, var):
        """
            Calculate only incremental value, do not update.
        :param var: variable or placeholder
        :return: delta w
        """
        pass
