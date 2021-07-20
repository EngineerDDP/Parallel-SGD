from abc import ABCMeta, abstractmethod
from typing import List

import rpc.communication as communication


class IExecutable(metaclass=ABCMeta):

    @abstractmethod
    def requests(self) -> List[object]:
        """
            Requested types
        return: list of objects.
        """
        pass

    @abstractmethod
    def satisfy(self, reply) -> List[object]:
        """
            Satisfy requested data.
        :return: list of Req(Enum) contains requests which cannot be satisfied.
        """
        pass

    @abstractmethod
    def start(self, com: communication.Communication):
        """
            Do the job.
        """
        pass

    @abstractmethod
    def ready(self) -> bool:
        """
            Is the rpc ready for the job.
        """
        pass

    @abstractmethod
    def done(self) -> bool:
        """
            Is job done?
        """
        pass

    @abstractmethod
    def trace_files(self) -> List[str]:
        """
            Return the filename list or executing trace.
        """
        pass
