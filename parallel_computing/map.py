from typing import Type, Iterable

from parallel_computing.interface import IMap
from models import ClassSerializer
from models import IReplyPackage


class MapOperator(IReplyPackage):

    def __init__(self, uuid_before: str, uuid_after: str, map_func: Type[IMap]):
        self.__before = uuid_before
        self.__after = uuid_after
        self.__map_func: ClassSerializer[IMap] = ClassSerializer(map_func)

    def restore(self) -> None:
        self.__map_func.restore()

    def uuid(self):
        return self.__before

    def new_uuid(self):
        return self.__after

    def do(self, data: Iterable[object]):
        return [self.__map_func().call(item) for item in data]
