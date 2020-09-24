from abc import ABCMeta, abstractmethod


class ITransformer(metaclass=ABCMeta):

    @property
    @abstractmethod
    def params(self):
        """
            Parameters used for rebuild this transformer.
        """
        pass

    @abstractmethod
    def __call__(self, train_x, train_y, test_x, test_y) -> tuple:
        """
            Transform dataset.
        :return: new dataset, train_x, train_y, test_x, test_y
        """
        pass

    @abstractmethod
    def add(self, trans):
        pass
