import numpy as np

from parallel_sgd.profiles.interface import IBatchIter


class BatchIter(IBatchIter):

    def __init__(self, block_size, block_count):
        # save make sure each block has same batch size
        self.__block_size = block_size
        # get default division
        self.__splitters = np.arange(0, block_count * block_size + 1, block_size)
        # store batch size
        self.__batch_size = self.__block_size * block_count

    @property
    def batch_size(self):
        return self.__batch_size

    def adjust_ratio(self, block_size_ratio:[list, tuple, np.ndarray]):
        # validate the proportion
        assert np.sum(block_size_ratio) <= 1, \
            "Block size ratio requires an array with sum equals 1, but {} was given.".format(np.sum(block_size_ratio))
        # update block size with given proportion
        block_len = np.floor(np.asarray(block_size_ratio) * self.batch_size)

        # extend each blocks.
        for i in range(1, len(self.__splitters)):
            self.__splitters[i] = self.__splitters[i - 1] + block_len[i - 1]

    def iter(self, batch_no, block_no):
        _offset = batch_no * self.batch_size
        # return slice
        return slice(self.__splitters[block_no] + _offset, self.__splitters[block_no + 1] + _offset)
