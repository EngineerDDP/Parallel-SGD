from abc import ABCMeta, abstractmethod
from typing import List

from numpy import ndarray

from nn.data.interface import IDataFeeder
from profiles.interface import IBatchIter


class IPSGDBlockMgr(metaclass=ABCMeta):

    @property
    @abstractmethod
    def current_block_id(self):
        pass

    @property
    @abstractmethod
    def batch_id(self):
        pass

    @property
    @abstractmethod
    def end(self):
        pass


class PSGDBlockDataFeeder(IDataFeeder, IPSGDBlockMgr):

    def __init__(self, x: ndarray, y: ndarray, batch_iter: IBatchIter, block_ids: List[int]):
        self.__x: ndarray = x
        self.__y: ndarray = y
        self.__total_blocks: List[int] = block_ids
        self.__cur_block: int = self.__total_blocks[0]
        self.__iter: int = 0
        self.__batch_id: int = 0
        self.__end: bool = False
        assert batch_iter.batch_size > len(x), \
            "Number of input samples is too small. P-SGD requires {} at least.".format(batch_iter.batch_size)
        self.__batch_size: int = batch_iter.batch_size
        self.__batch_iter: IBatchIter = batch_iter
        self.__batches: int = len(x) // self.__batch_size

    @property
    def position(self):
        return self.__iter

    @property
    def batch_id(self):
        return self.__batch_id

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def end(self):
        return self.__end

    @property
    def length(self):
        return self.__total_blocks

    @property
    def current_block_id(self):
        return self.__cur_block

    def __iter__(self):
        for self.__batch_id in range(self.__batches):
            self.__end = False
            for b_id in self.__total_blocks:
                self.__cur_block = b_id
                self.__iter += 1
                self.__end = b_id == self.__total_blocks[-1]
                sli = self.__batch_iter.iter(self.__batch_id, b_id)
                part_x = self.__x[sli]
                part_y = self.__y[sli]
                yield part_x, part_y

    def __repr__(self):
        print(self.__str__())

    def __str__(self):
        return "<P-SGD data iterator, current batch: {} in block: {}, total: {}."\
            .format(self.__iter, self.__cur_block, self.__batches)
