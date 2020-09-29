from abc import abstractmethod

from dataset.transforms.interface import ITransformer


class AbsTransformer(ITransformer):

    def __init__(self):
        self.__next = None

    def add(self, transformer) -> ITransformer:
        transformer.__next = self
        return transformer

    def __repr__(self):
        cur = self.__next
        res = []
        while cur is not None:
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
