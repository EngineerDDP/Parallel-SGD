import numpy as np

from itertools import combinations
from profiles.blockassignment.interfaces import IBlockAssignment


class IIDBlockAssignment(IBlockAssignment):

    def __init__(self, node_count, redundancy):

        self.__block_count = int(np.math.factorial(node_count) / (
                    np.math.factorial(redundancy) * np.math.factorial(node_count - redundancy)))

        self.__block_ids = list(range(self.__block_count))
        node_ids = list(range(node_count))

        self.__block_2_node = list(combinations(node_ids, redundancy))
        self.__node_2_block = [[] for _ in node_ids]

        block_id = 0

        for nodes in self.__block_2_node:
            for node in nodes:
                self.__node_2_block[node].append(block_id)
            block_id += 1

    @property
    def block_2_node(self):
        return self.__block_2_node

    @property
    def node_2_block(self):
        return self.__node_2_block

    @property
    def block_count(self):
        return self.__block_count

    @property
    def blocks(self):
        return self.__block_ids
