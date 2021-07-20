from abc import ABCMeta, abstractmethod


class IRequestPackage(metaclass=ABCMeta):

    @abstractmethod
    def content(self) -> object:
        pass


class IReplyPackage(metaclass=ABCMeta):

    @abstractmethod
    def restore(self) -> None:
        pass

