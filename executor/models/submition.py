from typing import Set

from executor.models.dynamic_modules import ClassSerializer
from executor.models.interface import IReplyPackage


class SubmitJob(IReplyPackage):

    def __init__(self, nodes: set, eta_waiting_time: int, exe: type):
        self.__nodes = nodes
        self.__eta_wait = eta_waiting_time
        self.__cls: [ClassSerializer, type] = ClassSerializer(exe)

    def restore(self) -> None:
        self.__cls = self.__cls.restore()

    @property
    def executioner(self) -> [type]:
        return self.__cls

    @property
    def work_group(self) -> Set[int]:
        return self.__nodes

    @property
    def waiting_time(self) -> [float, int]:
        return self.__eta_wait


class RequestWorkingLog:

    def __init__(self):
        pass
