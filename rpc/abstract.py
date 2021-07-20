from abc import ABCMeta, abstractmethod

import rpc.communication
import rpc.interface


class AbsExecutable(rpc.interface.IExecutable, metaclass=ABCMeta):

    def __init__(self, node_id: int, working_group: set, initializer_id: int):
        self.__node_id = node_id
        self.__working_group = working_group
        self.__initializer_id = initializer_id

    @property
    def node_id(self):
        return self.__node_id

    @property
    def group(self):
        return self.__working_group

    @property
    def initializer_id(self):
        return self.__initializer_id

    @abstractmethod
    def requests(self) -> list:
        """
            Requested types
        return: list of Req(Enum) objects.
        """
        pass

    @abstractmethod
    def satisfy(self, reply: list) -> list:
        """
            Satisfy requested data.
        :return: list of Req(Enum) contains requests which cannot be satisfied.
        """
        pass

    @abstractmethod
    def start(self, com: rpc.communication.Communication) -> object:
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
    def trace_files(self) -> list:
        """
            Return the filename list or executing trace.
        """
        pass


class AbsSimpleExecutor(AbsExecutable):

    def __init__(self, node_id: int, working_group: set, initializer_id: int = -1):
        super().__init__(node_id, working_group, initializer_id)
        self.__done = False

    def requests(self) -> list:
        return []

    def satisfy(self, reply: list) -> list:
        return []

    def ready(self) -> bool:
        return True

    def done(self) -> bool:
        return self.__done

    def start(self, com: rpc.communication.Communication) -> object:
        result = self.run(com)
        self.__done = True
        return result

    @abstractmethod
    def run(self, com: rpc.communication.Communication) -> object:
        pass

    def trace_files(self) -> list:
        return []
