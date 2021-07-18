from abc import ABCMeta, abstractmethod
# from multiprocessing import Queue
from typing import Sequence, Tuple, Optional, Union
from uuid import uuid4

import constants


class NodeAssignment:

    def __init__(self):
        self.__nodes = []
        self.__unique = set()
        self.__uuid = uuid4()

    @property
    def uuid(self):
        return self.__uuid

    def add(self, id, addr):
        assert id not in self.__unique, "Assigned id has been used."
        assert id != constants.Promoter_ID, "Id ({}) is reserved for internal usage.".format(constants.Promoter_ID)
        self.__nodes.append((id, addr))
        self.__unique.add(id)

    def __iter__(self):
        for _id, addr in self.__nodes:
            yield _id, addr

    def __len__(self):
        return len(self.__unique)

    def __repr__(self):
        str = "({}) nodes has been assigned:\n\t\t-->".format(len(self))
        str += '\n\t\t-->'.join(["ID:{:>4d}\t\tAddress:  {}".format(id, addr) for id, addr in self])
        return str

    def __str__(self):
        return self.__repr__()


class INodeRegister(metaclass=ABCMeta):

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def register(self, id_self, content_package, io_event):
        """
            Register all workers
        :param id_self: id of current worker
        :param content_package: content package that contains address and uuid of all workers
        :return: None
        """
        pass

    @abstractmethod
    def check(self):
        pass

    @abstractmethod
    def identify(self, id, uuid, con):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    @abstractmethod
    def working_port(self):
        pass


class AbsCommunicationProcess(metaclass=ABCMeta):

    @property
    @abstractmethod
    def bytes_sent(self):
        pass

    @property
    @abstractmethod
    def bytes_read(self):
        pass

    @abstractmethod
    def start(self):
        """
        Start handling traffics from networks
        :return: None
        """
        pass

    @abstractmethod
    def get(self, blocking: bool, timeout: int) -> Tuple[int, object]:
        """
        Get a object

        If optional args 'block' is true and 'timeout' is None (the default),
        block if necessary until an item is available. If 'timeout' is
        a non-negative number, it blocks at most 'timeout' seconds and raises
        the Empty exception if no item was available within that time.
        Otherwise ('block' is false), return an item if one is immediately
        available, else raise the Empty exception ('timeout' is ignored
        in that case).

        :param blocking:    Dose this operation block the running process
        :param timeout:     How long would you like to wait
        :return: Int and Object, int represents the sender's id, object is the object that received.
        """
        pass

    @abstractmethod
    def put(self, target: Union[Sequence[int], int], obj: object, blocking: bool, timeout: int):
        """
        Put a object

        If optional args 'block' is true and 'timeout' is None (the default),
        block if necessary until an item is available. If 'timeout' is
        a non-negative number, it blocks at most 'timeout' seconds and raises
        the Empty exception if no item was available within that time.
        Otherwise ('block' is false), return an item if one is immediately
        available, else raise the Empty exception ('timeout' is ignored
        in that case).

        :param target:      Send to who
        :param obj:         Send what
        :param blocking:    Dose this operation block the running process
        :param timeout:     How long would you like to wait
        :return: Bool
        """
        pass

    @abstractmethod
    def flush_data_and_quit(self):
        """
        Quit from distributed network and flush all unsent data.
        :return: None
        """
        pass

    @abstractmethod
    def force_quit(self):
        """
        Quit from distributed network instantly, discard all data.
        :return:
        """
        pass

    @abstractmethod
    def has_quit(self) -> bool:
        """
        Check if has quited from network.
        :return: True or False
        """
        pass

    @property
    @abstractmethod
    def available_nodes(self):
        pass

    @property
    @abstractmethod
    def node_id(self):
        pass

    @property
    @abstractmethod
    def nodes(self):
        pass


class ICommunicationController(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def __enter__(self) -> "ICommunicationController":
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    @abstractmethod
    def Com(self) -> AbsCommunicationProcess:
        pass

    @property
    @abstractmethod
    def Node_Id(self) -> int:
        pass

    @abstractmethod
    def get_one(self, blocking=True, timeout: int = None) -> Tuple[Optional[int], object]:
        """
            Get one json like object from target nodes.
        :param blocking: Dose this operation block the running process
        :param timeout: How long would you like to wait
        :return: a tuple, which first element is the sender id, second element is the object.
                 (None, None) was returned if data wasn't available.
        """
        pass

    @abstractmethod
    def send_one(self, target, dic, timeout: int = None) -> bool:
        """
            send one json like object to target nodes
        :param target: target node list, must be a list : list[int]
        :param dic: json like object : encode
        :param timeout: How long would you like to wait
        :return: True or False, True means your data has been scheduled and ready to be sent, otherwise not.
        """
        pass

    @property
    @abstractmethod
    def available_clients(self):
        pass

    @property
    @abstractmethod
    def available_clients_count(self):
        pass

    @abstractmethod
    def close(self, force: bool = False, timeout: int = 20):
        """
            Stop communicating with remote nodes.
        :param force: force quit, discard all data
        :param timeout: if (@param force) is True, ignore this parameter,
                        otherwise wait (@param timeout) seconds and do force quit.
        :return: None
        """
        pass

    @abstractmethod
    def is_closed(self):
        """
            Check if the communication thread is already closed.
        :return: True if closed, False if still running.
        """
        pass


class IPromoter(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, para: NodeAssignment) -> AbsCommunicationProcess:
        pass
