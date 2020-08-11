import numpy as np

from profiles.blockassignment.idependent import IIDBlockAssignment


class Batch:

    def __init__(self, batch_size, block_count):
        # save make sure each block has same batch size
        self.batch_size = batch_size * block_count
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


class GlobalSettings:
    __setting = None

    def __init__(self, n=4, r=2, batch_size=96, assignment_type=IIDBlockAssignment):
        self.redundancy = r
        self.node_count = n

        self.block_assignment = assignment_type(n, r)
        self.batch = Batch(batch_size, self.block_assignment.block_count)
        self.nodes = set(range(self.node_count))

    @staticmethod
    def get_default():
        if GlobalSettings.__setting is None:
            GlobalSettings.__setting = GlobalSettings()
        return GlobalSettings.__setting

    @staticmethod
    def clear_default():
        GlobalSettings.__setting = None

    @staticmethod
    def set_default(n, r, b, assignment):
        """
            Set global setting, if not settled
        :param n: number of workers
        :param r: redundancy
        :param b: batch size globally,
                  when set to 12 with 3 nodes, the the local batch size on each node is 4.
        :param assignment: blockassignment class
        :return:
        """
        if GlobalSettings.__setting is None:
            GlobalSettings.__setting = GlobalSettings(n, r, b, assignment)
