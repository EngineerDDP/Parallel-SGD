import numpy as np

from itertools import combinations, permutations


class Batch:

    def __init__(self, batch_size, block_count):
        # get default division
        self.Splitters = np.floor(np.linspace(0, batch_size, block_count + 1))

        # save total batch size
        self.Batch_Size = batch_size

    def updateBlockWithRatio(self, block_size_ratio):
        # update block size with occupation ratio
        block_start = 0
        block_len = np.floor(np.asarray(block_size_ratio) * self.Batch_Size)

        i = 1

        # update slice
        while i < len(self.Splitters):
            self.Splitters[i] = self.Splitters[i - 1] + block_len[i - 1]

            i += 1

    def getSlice(self, offset, block_no):
        # return slice
        return slice(int(self.Splitters[block_no] + offset), int(self.Splitters[block_no + 1] + offset))

    def getSlicelen(self, block_no):
        sli = self.getSlice(0, block_no)
        return sli.stop - sli.start


class BlockAssignment:

    def __init__(self, global_setting):

        block_ids = list(range(global_setting.BlockCount))
        node_ids = list(range(global_setting.NodeCount))

        self.Block2Node = list(combinations(node_ids, global_setting.Redundancy))
        self.Node2Block = [[] for i in node_ids]

        block_id = 0

        for nodes in self.Block2Node:
            for node in nodes:
                self.Node2Block[node].append(block_id)
            block_id += 1


class GlobalSettings:
    __setting = None

    def __init__(self, n=4, r=2, batch_size=96):
        self.Redundancy = r
        self.NodeCount = n

        self.BlockCount = int(np.math.factorial(self.NodeCount) / (
                    np.math.factorial(self.Redundancy) * np.math.factorial(self.NodeCount - self.Redundancy)))

        self.Batch = Batch(batch_size, self.BlockCount)
        self.Nodes = set(range(self.NodeCount))
        self.Blocks = set(range(self.BlockCount))
        self.BlockAssignment = BlockAssignment(self)

    @staticmethod
    def get_default():
        if GlobalSettings.__setting is None:
            GlobalSettings.__setting = GlobalSettings()
        return GlobalSettings.__setting

    @staticmethod
    def set_default(n, r, b):
        GlobalSettings.__setting = GlobalSettings(n, r, b)
        return 
