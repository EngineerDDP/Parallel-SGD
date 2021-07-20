from enum import Enum
from typing import Iterable

Worker_File = "executors.json"


class Commands(Enum):
    Close = "RDD_Close"


class Drop:

    def __init__(self, uuid: str):
        self.uuid = uuid


class Add:

    def __init__(self, uuid: str, content: Iterable[object]):
        # TODO:
        # Implements operation
        pass
