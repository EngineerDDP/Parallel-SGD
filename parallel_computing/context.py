import json
from typing import Iterable

from parallel_computing.constants import Worker_File
from parallel_computing.controller import RDDController
from parallel_computing.interface import IContext, IRDD
from parallel_computing.worker import RDDNode
from network import NodeAssignment, Request
from roles.coordinator import Coordinator
from log import MutePrinter


class Context(IContext):

    def __init__(self):
        nodes = NodeAssignment()

        id_start = 0
        with open(Worker_File, 'r') as file:
            for addr in json.load(file):
                nodes.add(id_start, addr)
                id_start += 1

        req = Request()
        com = req.request(nodes)
        coor = Coordinator(com, estimate_bandwidth=10, logger=MutePrinter("FSpark_Ctx", log_to_file=True))
        coor.submit_group(RDDNode, com.available_clients)

        self.__rdd_ctx = RDDController(com)

    def parallel(self, data: Iterable[object]) -> IRDD:
        # TODO:
        # make new rdd
        pass
