from typing import Type

from parallel_computing.interface import IRDD, IController, IMap, IReduce


class RDD(IRDD):

    def __init__(self, uuid: str, access: IController):
        self.__uuid = uuid
        self.__access = access

    def map(self, map_func: Type[IMap]) -> "IRDD":
        return RDD(self.__access.transform(self.__uuid, map_func), self.__access)

    def reduce(self, reduce_func: Type[IReduce]) -> object:
        data_ctx = self.__access.reduce(self.__uuid)
        if len(data_ctx) == 0:
            return None
        if len(data_ctx) == 1:
            return data_ctx[0]

        aggregate = reduce_func()
        result = aggregate.call(data_ctx[0], data_ctx[1])

        for i in range(2, len(data_ctx)):
            result = aggregate.call(result, data_ctx[i])

        return result

    def __del__(self):
        self.__access.drop(self.__uuid)
