from abc import abstractmethod, ABCMeta, abstractproperty


class IBlockAssignment(metaclass=ABCMeta):

    @property
    @abstractmethod
    def block_2_node(self):
        """
            block to node hash, to identify which nodes has the key block.
        :return: dict or list, like {block : [node]}
        """
        pass

    @property
    @abstractmethod
    def node_2_block(self):
        """
            node to block hash, to identify which block were sent to key node.
        :return: dict or list, like {node : [block]}
        """
        pass

    @property
    @abstractmethod
    def block_count(self):
        """
            total blocks
        :return: int
        """
        pass

    @property
    @abstractmethod
    def blocks(self):
        """
            id of each block
        :return: [0, 1, 2, ... ,block_count()-1]
        """
        pass

