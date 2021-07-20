from abc import ABCMeta, abstractmethod
from typing import Type, Iterable, Sequence


class IMap(metaclass=ABCMeta):

    @abstractmethod
    def call(self, x):
        pass


class IReduce(metaclass=ABCMeta):

    @abstractmethod
    def call(self, x, y):
        pass


class IRDD(metaclass=ABCMeta):

    @abstractmethod
    def map(self, map_func: Type[IMap]) -> "IRDD":
        pass

    @abstractmethod
    def reduce(self, reduce_func: Type[IReduce]) -> object:
        pass


class IContext(metaclass=ABCMeta):

    @abstractmethod
    def parallel(self, data: Iterable[object]) -> IRDD:
        pass


class IController(metaclass=ABCMeta):

    @abstractmethod
    def drop(self, uuid: str):
        pass

    @abstractmethod
    def transform(self, uuid: str, function: Type[IMap]) -> str:
        pass

    @abstractmethod
    def reduce(self, uuid: str) -> Sequence[object]:
        pass
