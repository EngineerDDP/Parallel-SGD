from abc import ABCMeta, abstractmethod
from typing import List, Set


class ISetting(metaclass=ABCMeta):

    @property
    @abstractmethod
    def redundancy(self) -> int:
        pass

    @property
    @abstractmethod
    def node_count(self) -> int:
        pass

    @property
    @abstractmethod
    def nodes(self) -> List[int]:
        """
            List of node id.
        :return: list
        """
        pass

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

    @abstractmethod
    def get_adversary(self, block_id: int) -> Set[int]:
        pass
