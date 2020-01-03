from abc import abstractmethod, ABCMeta, abstractproperty


class IBlockAssignment(metaclass=ABCMeta):

    @property
    @abstractmethod
    def block_2_node(self):
        pass

    @property
    @abstractmethod
    def node_2_block(self):
        pass

    @property
    @abstractmethod
    def block_count(self):
        pass

    @property
    @abstractmethod
    def blocks(self):
        pass

