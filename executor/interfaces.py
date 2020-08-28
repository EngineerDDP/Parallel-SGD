from abc import ABCMeta, abstractmethod

from models.local.__init__ import IServerModel
from dataset.interfaces import IDataset
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
    def add_info(self, obj:IServerModel):
        """
            Add essential information.
        """
        pass

    @abstractmethod
    def add_data(self, obj:IDataset):
        """
            Add dataset reference
        """
        pass

    @abstractmethod
    def add_setting(self, obj:Settings):
        """
            Add settings.
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