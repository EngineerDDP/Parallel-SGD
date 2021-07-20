import json

from constants import Parameter_Server
from network import NodeAssignment


def parse_worker(worker_cnt: int, ps: bool = False, filename: str = "worker.json"):
    nodes = NodeAssignment()
    with open(filename, 'r') as file:
        data = json.load(file)
        if ps:
            nodes.add(Parameter_Server, data["PS"])
        for i in range(worker_cnt):
            nodes.add(i, data["Worker"][i])

    return nodes
