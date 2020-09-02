from abc import ABCMeta, abstractmethod

from dataset.interfaces import IDataset
from models.local import IServerModel
from network.interfaces import ICommunication_Controller
from profiles import Settings


class IExecutor(metaclass=ABCMeta):

    def __init__(self, node_id:int, VLAN_offset:int):
        self.__node_id = node_id
        self.__vlan_offset = VLAN_offset

    @property
    def node_id(self):
        return self.__node_id

    @property
    def group_offset(self):
        return self.__vlan_offset

    @abstractmethod
    def requests(self) -> list:
        """
            Requested types
        return: list of Req(Enum) objects.
        """
        pass

    @abstractmethod
    def satisfy(self, reply:list) -> list:
        """
            Satisfy requested data.
        :return: list of Req(Enum) contains requests which cannot be satisfied.
        """
        pass

    @abstractmethod
    def run(self, com:ICommunication_Controller) -> None:
        """
            Do the job.
        """
        pass

    @abstractmethod
    def ready(self) -> bool:
        """
            Is the executor ready for the job.
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


class AbsSimpleExecutor(IExecutor):

    def add_info(self, obj: IServerModel):
        pass

    def add_data(self, obj: IDataset):
        pass

    def add_setting(self, obj: Settings):
        pass

    def ready(self) -> bool:
        return True

    def done(self) -> bool:
        return True

    @abstractmethod
    def run(self, com: ICommunication_Controller) -> None:
        pass

    @abstractmethod
    def trace_files(self) -> list:
        pass