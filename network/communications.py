import socket
import select

from time import sleep
from constants import Initialization_Server

from network.agreements import *
from network.interfaces import IWorker_Register, ICommunication_Process
from network.serialization import unpack


class Worker_Communication_Constructor:
    """
        Communication constructor
        Factory class for build class Com
    """

    def __init__(self, server, port, worker_register: IWorker_Register):
        """
            Typo server address
        :param server:
        """
        self.Server = server
        self.Port = port
        # setup job listener
        self.__bind_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # using Non-blocking IO
        self.__bind_listener.setblocking(False)
        # bind address
        self.__bind_listener.bind((self.Server, self.Port))
        self.__id_register = worker_register

    def buildCom(self):
        """
            Non-blocking IO for register this slave com to a specified job.
            Connection will be established while all connections between slaves were established.
        """
        # temporary register
        _tmp_register_ref_table = [self.__bind_listener]
        # start listening
        self.__bind_listener.listen(4)

        # wait for job submission
        while not self.__id_register.check():
            readable, writable, exp = select.select(_tmp_register_ref_table, [], _tmp_register_ref_table)
            for io_event in readable:
                if io_event is self.__bind_listener:
                    # accept connection, could be submitter or other slaves
                    con, client_address = io_event.accept()
                    # add to identified list
                    _tmp_register_ref_table.append(con)
                    con.setblocking(False)

                else:
                    data = unpack(io_event)
                    # message from submitter
                    if data[Key.Type] == Type_Val.Submission:
                        self.__id_register.register(data[Key.To], data[Key.Content])
                        self.__id_register.put(Initialization_Server, io_event)
                    # message from other worker
                    if data[Key.Type] == Type_Val.WorkerReports:
                        self.__id_register.identify(data[Key.From], data[Key.Content], io_event)

        return self.__id_register


class Communication_Controller:

    def __init__(self, com: ICommunication_Process):
        """
            Prepare communication module for connection.
            Change CommunicationController.static_server_address and CommunicationController.static_server_port
            before initializing this class.
        """
        self.com = com
        self.Node_ID = com.node_id()

    def establish_communication(self):
        """
            Establish connection.
        :return: None
        """
        self.com.start()

    def get_one(self):
        """
            Get one json like object from target nodes.
        :return: a tuple, which first element is the sender id, second element is the json object.
        """
        return self.com.recv_que.get()

    def send_one(self, target, dic):
        """
            send one json like object to target nodes
        :param target: target node list, must be a list : list[int]
        :param dic: json like object : encode
        :return: None
        """
        self.com.send_que.put((target, dic))

        return None

    def close(self):
        """
            Stop communicating with remote nodes.
        :return: None
        """
        self.com.close()
        sleep(1)
        self.com.terminate()

    def is_closed(self):
        """
            Check if the communication thread is already closed.
        :return: True if closed, False if still running.
        """
        return self.com.Exit.value is True


if __name__ == "__main__":
    pass
