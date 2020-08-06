import queue
import socket
import select

from time import sleep
from utils.constants import Initialization_Server

from network.agreements import *
from network.interfaces import IWorker_Register, ICommunication_Process, ICommunication_Controller
from network.serialization import Buffer


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
        self.__id_register = worker_register

    def buildCom(self):
        """
            Non-blocking IO for register this slave com to a specified job.
            Connection will be established while all connections between slaves were established.
        """
        # release while leave active area
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as __bind_listener:
            # using Non-blocking IO
            __bind_listener.setblocking(False)
            # bind address
            __bind_listener.bind((self.Server, self.Port))
            # start listening
            __bind_listener.listen(4)
            # temporary register
            _tmp_register_ref_table = [__bind_listener]
            _tmp_buffer_recv = {__bind_listener: Buffer()}

            # wait for job submission
            while not self.__id_register.check():
                readable, writable, exp = select.select(_tmp_register_ref_table, [], _tmp_register_ref_table)
                for io_event in readable:
                    if io_event is __bind_listener:
                        # accept connection, could be submitter or other slaves
                        con, client_address = io_event.accept()
                        # add to identified list
                        _tmp_register_ref_table.append(con)
                        con.setblocking(False)

                    else:
                        buf = _tmp_buffer_recv.get(io_event, Buffer())
                        buf.recv(io_event)
                        _tmp_buffer_recv[io_event] = buf

                        if buf.is_ready():
                            data = buf.get_content()
                            # message from submitter
                            if data[Key.Type] == Type_Val.Submission:
                                self.__id_register.register(data[Key.To], data[Key.Content], io_event)
                            # message from other worker
                            elif data[Key.Type] == Type_Val.WorkerReports:
                                self.__id_register.identify(data[Key.From], data[Key.Content], io_event)
                                print('acc worker')

        return self.__id_register


class Communication_Controller(ICommunication_Controller):

    def __init__(self, com: ICommunication_Process):
        """
            Prepare communication module for connection.
            Change CommunicationController.static_server_address and CommunicationController.static_server_port
            before initializing this class.
        """
        super().__init__()
        self.__com = com

    @property
    def Node_Id(self):
        return self.__com.node_id()

    def establish_communication(self):
        """
            Establish connection.
        :return: None
        """
        self.__com.start()

    def get_one(self, blocking=True):
        """
            Get one json like object from target nodes.
        :return: a tuple, which first element is the sender id, second element is the json object.
        """
        if self.is_closed():
            raise OSError('Connection has already been closed.')
        if self.__com.recv_que.empty() and not blocking:
            return None
        while self.__com.is_alive():
            try:
                return self.__com.recv_que.get(timeout=1)
            except queue.Empty:
                continue
        raise OSError('Connection is closed.')

    def send_one(self, target, dic):
        """
            send one json like object to target nodes
        :param target: target node list, must be a list : list[int]
        :param dic: json like object : encode
        :return: None
        """
        if self.is_closed():
            raise OSError('Connection has already been closed.')
        if isinstance(target, list):
            self.__com.send_que.put((target, dic))
        else:
            self.__com.send_que.put(([target], dic))

        return None

    def available_clients(self):
        return self.__com.nodes()

    def close(self):
        """
            Stop communicating with remote nodes.
        :return: None
        """
        self.__com.close()
        while self.__com.is_alive():
            sleep(1)
        self.__com.terminate()

    def is_closed(self):
        """
            Check if the communication thread is already closed.
        :return: True if closed, False if still running.
        """
        return not self.__com.is_alive() and self.__com.recv_que.empty()


def get_repr():
    return socket.gethostbyname(socket.gethostname())

if __name__ == "__main__":
    pass
