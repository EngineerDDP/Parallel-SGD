import numpy as np

from profiles.blockassignment.interfaces import IBlockAssignment


class DuplicateAssignment(IBlockAssignment):
    """
        Full duplicated block assignment strategy.
        Every r nodes have the same blocks.
    """

    def __init__(self, node_count, redundancy):

        self.__block_count = node_count / redundancy
        self.__block_ids = list(range(self.__block_count))

        node_ids = np.arange(0, node_count, 1)
        self.__block_2_node = np.split(node_ids, self.__block_count)
        self.__node_2_block = [[block_id for _ in range(redundancy)] for block_id in self.__block_ids]

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

