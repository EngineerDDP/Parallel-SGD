import numpy as np

from profiles.blockassignment.interfaces import IBlockAssignment
from profiles.blockassignment.idependent import IIDBlockAssignment


class Batch:

    def __init__(self, batch_size, block_count):
        # save make sure each block has same batch size
        self.batch_size = batch_size
        # get default division
        self.Splitters = np.floor(np.linspace(0, self.batch_size, block_count + 1))

    def update_block_with_ratio(self, block_size_ratio):
        # update block size with occupation ratio
        block_start = 0
        block_len = np.floor(np.asarray(block_size_ratio) * self.batch_size)

        i = 1
        # update slice
        while i < len(self.Splitters):
            self.Splitters[i] = self.Splitters[i - 1] + block_len[i - 1]
            i += 1

    def get_slice(self, offset, block_no):
        # return slice
        return slice(int(self.Splitters[block_no] + offset), int(self.Splitters[block_no + 1] + offset))

    def get_slice_len(self, block_no):
        sli = self.get_slice(0, block_no)
        return sli.stop - sli.start


class Settings:

    def __init__(self, n, r, batch_size, assignment:IBlockAssignment=None):
        self.__redundancy = r
        self.__node_count = n

        if isinstance(assignment, IBlockAssignment):
            self.__block_assignment = assignment
        else:
            self.__block_assignment = IIDBlockAssignment(n, r)
        self.__batch = Batch(batch_size, self.block_assignment.block_count)
        self.__nodes = set(range(self.node_count))

    @property
    def redundancy(self) -> int:
        return self.__redundancy

    @property
    def node_count(self) -> int:
        return self.__node_count

    @property
    def block_assignment(self) -> IBlockAssignment:
        return self.__block_assignment

    @property
    def batch(self) -> Batch:
        return self.__batch

    @property
    def nodes(self) -> set:
        return self.__nodes
