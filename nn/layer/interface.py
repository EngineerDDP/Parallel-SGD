from abc import ABCMeta, abstractmethod


class ILazyInitialization(metaclass=ABCMeta):

    @abstractmethod
    def set_input(self, op):
        pass

    @abstractmethod
    def initialize_parameters(self, x) -> None:
        """
            Initialize parameters with given input_ref (x)
        :param x: ndarray
        """
        pass
