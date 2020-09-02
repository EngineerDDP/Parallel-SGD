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

class AbsTransformer(ITransformer):

    def __init__(self):
        self.__next = None

    def add(self, transformer):
        transformer.__next = self.__next
        self.__next = transformer

    def __repr__(self):
        cur = self.__next
        res = []
        while cur != None:
            res.append(str(cur))
            cur = cur.__next
        return "<Transformation: ({})>".format(','.join(res))

    def __call__(self, train_x, train_y, test_x, test_y) -> tuple:
        if self.__next is not None:
            train_x, train_y, test_x, test_y = self.__next(train_x, train_y, test_x, test_y)

        return self.run(train_x, train_y, test_x, test_y)

    @abstractmethod
    def run(self, train_x, train_y, test_x, test_y):
        pass


class TransformerList(AbsTransformer):

    def __init__(self):
        super().__init__()

    def params(self):
        return tuple()

    def run(self, train_x, train_y, test_x, test_y):
        return train_x, train_y, test_x, test_y