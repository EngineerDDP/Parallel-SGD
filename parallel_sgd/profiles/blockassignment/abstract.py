from abc import abstractmethod
from typing import Set, List

from parallel_sgd.profiles.blockassignment.interface import ISetting


class AbsBlockAssignment(ISetting):

    def __init__(self, n, r):
        self.__n = n
        self.__r = r
        self.__node_ids = list(range(n))

    @property
    def redundancy(self) -> int:
        return self.__r

    @property
    def node_count(self) -> int:
        return self.__n

    @property
    def nodes(self) -> List[int]:
        """
            List of node id.
        :return: list
        """
        return self.__node_ids

    @property
    @abstractmethod
    def block_2_node(self) -> List[List[int]]:
        """
            block to node hash, to identify which nodes has the key block.
        :return: list, like [block : [node]]
        """
        pass

    @property
    @abstractmethod
    def node_2_block(self) -> List[List[int]]:
        """
            node to block hash, to identify which block were sent to key node.
        :return: list, like [node : [block]]
        """
        pass

    @property
    @abstractmethod
    def block_count(self) -> int:
        """
            total blocks
        :return: int
        """
        pass

    @property
    @abstractmethod
    def blocks(self) -> List[int]:
        """
            id of each block
        :return: [0, 1, 2, ... ,block_count()-1]
        """
        pass

    def get_adversary(self, block_id: int) -> Set[int]:
        return set(self.nodes) - set(self.block_2_node[block_id])
