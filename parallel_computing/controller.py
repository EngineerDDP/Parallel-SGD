from typing import Sequence, Type
from uuid import uuid4

from rpc.communication import Communication
from parallel_computing.constants import Drop
from parallel_computing.interface import IController, IMap
from parallel_computing.map import MapOperator
from parallel_computing.reduce import ReduceOperator


class RDDController(IController):

    def __init__(self, com: Communication):
        self.__com = com
        self.__rdd_register = set()

    def transform(self, uuid: str, function: Type[IMap]) -> str:
        op = MapOperator(uuid, str(uuid4()), function)
        self.__com.send_one(self.__com.available_clients, op)
        return op.new_uuid()

    def drop(self, uuid: str):
        self.__com.send_one(self.__com.available_clients, Drop(uuid))

    def reduce(self, uuid: str) -> Sequence[object]:
        all_nodes = self.__com.available_clients
        self.__com.send_one(self.__com.available_clients, ReduceOperator(uuid))
        accepted_nodes = {}

        while len(accepted_nodes) != len(all_nodes):
            id_from, data = self.__com.get_one()
            accepted_nodes[id_from] = data

        result = []
        for item in accepted_nodes.values():
            result.extend(item)
        return result
