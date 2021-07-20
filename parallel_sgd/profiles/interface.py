from abc import ABCMeta, abstractmethod

from numpy import ndarray


class IBatchIter(metaclass=ABCMeta):

    @property
    @abstractmethod
    def batch_size(self):
        pass

    @abstractmethod
    def iter(self, batch_no:int, block_no:int) -> slice:
        pass

    @abstractmethod
    def adjust_ratio(self, block_size_ratio:[list, tuple, ndarray]):
        pass
