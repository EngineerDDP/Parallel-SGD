from typing import Tuple
from numpy import ndarray
from abc import ABCMeta, abstractmethod

from nn.data.interface import IDataFeeder


class IPSGDBlockMgr(metaclass=ABCMeta):

    @property
    @abstractmethod
    def current_block_id(self):
        pass


class PSGDBlockDataFeeder(IDataFeeder, IPSGDBlockMgr):

    def __init__(self, x: Tuple[ndarray], y: Tuple[ndarray], batch_size:int, block_ids:Tuple[int]):
        self.__iter_x:Tuple[ndarray] = x
        self.__iter_y:Tuple[ndarray] = y
        self.__total_blocks:Tuple[int] = block_ids
        self.__iter:int = 0
        self.__batches:int = 0
        self.__block:int = 0
        self.__batch_size:int = min(batch_size, len(x))
        self.__batches:int = len(x) // batch_size

    @property
    def position(self):
        return self.__iter

    @property
    def length(self):
        return self.__total_blocks

    @property
    def current_block_id(self):
        return self.__block

    def __iter__(self):
        for self.__iter in range(self.__batches):
            for b_id, b_x, b_y in zip(self.__total_blocks, self.__iter_x, self.__iter_y):
                start = self.__iter * self.__batch_size % (len(self.__iter_x) - self.__batch_size + 1)
                end = start + self.__batch_size
                self.__block = b_id
                part_x = b_x[start:end]
                part_y = b_y[start:end]
                self.__iter += 1
                yield part_x, part_y

    def __repr__(self):
        print(self.__str__())

    def __str__(self):
        return "<P-SGD data iterator, current batch: {} in block: {}, total: {}.".format(self.__iter, self.__block, self.__batches)