import numpy as np

from parallel_sgd.profiles.blockassignment.abstract import AbsBlockAssignment


class DuplicateAssignment(AbsBlockAssignment):
    """
        Full duplicated block assignment strategy.
        Every r nodes have the same blocks.
    """

    def __init__(self, node_count: int, redundancy: int):

        super().__init__(node_count, redundancy)
        self.__block_count = node_count // redundancy
        self.__block_ids = list(range(self.__block_count))

        self.__node_ids = np.arange(0, node_count, 1)
        self.__block_2_node = np.split(self.__node_ids, self.__block_count)
        self.__node_2_block = [[block_id] for block_id in self.__block_ids for _ in range(redundancy)]

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

