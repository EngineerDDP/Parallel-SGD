from abc import ABCMeta, abstractmethod, abstractproperty


class IDataFeeder(metaclass=ABCMeta):

    @property
    @abstractmethod
    def position(self):
        pass

    @property
    @abstractmethod
    def length(self):
        pass

    @property
    @abstractmethod
    def batch_size(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass
