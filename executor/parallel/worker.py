import time
from typing import List, Iterable, Dict

from executor.communication import Communication
from executor.interface import IExecutor
from executor.parallel.constants import Commands, Drop
from executor.parallel.map import MapOperator
from executor.parallel.reduce import ReduceOperator


class RDDNode(IExecutor):

    def __init__(self, node_id: int, working_group: set, initializer_id: int):
        self.__node_id = node_id
        self.__working_group = working_group
        self.__initializer_id = initializer_id
        # RDD Context
        self.__context: Dict[str, Iterable[object]] = dict()
        # RDD States
        self.__disposed: bool = False

    def requests(self) -> List[object]:
        return []

    def satisfy(self, reply: List[Iterable[object]]) -> List[object]:
        return []

    def __recv_pack(self, com: Communication):
        data = None
        id_from = None
        # requests with timeout check
        while data is None and self.__initializer_id in com.available_clients:
            id_from, data = com.get_one(blocking=False)
            time.sleep(0.01)
            # Assertion, this node count as one
        return id_from, data

    def start(self, com: Communication):
        """
            Start maintaining RDD
        :param com:
        :return: None
        """
        while not self.__disposed:

            id_from, ctx = RDDNode.__recv_pack(com)

            if id_from != self.__initializer_id:
                continue

            if isinstance(ctx, Commands):

                if ctx == Commands.Close:
                    self.__disposed = True

            elif isinstance(ctx, MapOperator):
                ctx.restore()
                before = self.__context.get(ctx.uuid(), [])
                self.__context[ctx.new_uuid()] = ctx.do(before)

            elif isinstance(ctx, ReduceOperator):
                com.send_one(self.__initializer_id, self.__context.get(ctx.uuid(), None))

            elif isinstance(ctx, Drop):
                del self.__context[ctx.uuid]

    def ready(self) -> bool:
        return self.__context is not None

    def done(self) -> bool:
        return self.__disposed

    def trace_files(self) -> List[str]:
        return []
