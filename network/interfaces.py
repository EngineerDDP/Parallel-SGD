from abc import ABCMeta, abstractmethod
from ctypes import c_bool
from multiprocessing import Queue
from multiprocessing import Value, Process
from uuid import uuid4


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
        self.__nodes.append((id, addr))

    def __iter__(self):
        for id, addr in self.__nodes:
            yield id, addr


class IWorker_Register(metaclass=ABCMeta):

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


class ICommunication_Process(Process, metaclass=ABCMeta):
    Circle_interval = 0.001

    def __init__(self, name: str):
        Process.__init__(self, name=name, daemon=True)
        self.__exit = Value(c_bool, 0)
        self.__recv_que = Queue(maxsize=24)
        self.__send_que = Queue(maxsize=24)

    @property
    @abstractmethod
    def bytes_sent(self):
        pass

    @property
    @abstractmethod
    def bytes_read(self):
        pass

    @property
    def recv_que(self):
        return self.__recv_que

    @property
    def send_que(self):
        return self.__send_que

    @property
    def Exit(self):
        return self.__exit.value

    @Exit.setter
    def Exit(self, value):
        self.__exit.value = value

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

    def closing(self):
        self.__exit.value = True


class ICommunication_Controller(metaclass=ABCMeta):

    def __init__(self):
        pass

    @property
    @abstractmethod
    def Com(self):
        pass

    @property
    @abstractmethod
    def Node_Id(self):
        pass

    @abstractmethod
    def get_one(self, blocking=True):
        """
            Get one json like object from target nodes.
        :return: a tuple, which first element is the sender id, second element is the json object.
        """
        pass

    @abstractmethod
    def send_one(self, target, dic):
        """
            send one json like object to target nodes
        :param target: target node list, must be a list : list[int]
        :param dic: json like object : encode
        :return: None
        """
        pass

    @abstractmethod
    def available_clients(self):
        pass

    @abstractmethod
    def close(self):
        """
            Stop communicating with remote nodes.
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
    def __call__(self, para: NodeAssignment) -> ICommunication_Process:
        pass
