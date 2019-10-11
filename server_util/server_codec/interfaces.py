from abc import ABCMeta, abstractmethod


class PACodec(metaclass=ABCMeta):

    @abstractmethod
    def update_blocks(self, content, tag):
        pass

    @abstractmethod
    def acquire_weights(self, tag):
        pass