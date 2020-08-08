from abc import ABCMeta, abstractmethod
from multiprocessing import Value, Process
from multiprocessing import Queue


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


class ICommunication_Process(Process, metaclass=ABCMeta):
    Circle_interval = 0.001

    def __init__(self, name: str):
        Process.__init__(self, name=name)
        self.__exit = Value('i', 0)
        self.__recv_que = Queue(maxsize=24)
        self.__send_que = Queue(maxsize=24)

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

    def close(self):
        self.__exit.value = True


class ICommunication_Controller(metaclass=ABCMeta):

    def __init__(self):
        pass

    @property
    @abstractmethod
    def Node_Id(self):
        pass

    @abstractmethod
    def establish_communication(self):
        """
            Establish connection.
        :return: None
        """
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