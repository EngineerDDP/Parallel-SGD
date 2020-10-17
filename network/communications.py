import queue
import socket
import select

from time import sleep

from network.agreements import *
from network.interfaces import IWorker_Register, AbsCommunicationProcess, ICommunication_Controller
from network.serialization import BufferReader


class Worker_Communication_Constructor:
    """
        Communication constructor
        Factory class for build class Com
    """

    def __init__(self, worker_register: IWorker_Register, server='0.0.0.0'):
        """
            Typo server address
        """
        self.__id_register = worker_register
        self.__server_addr = server

    def buildCom(self):
        """
            Non-blocking IO for register this slave com to a specified job.
            Connection will be established while all connections between slaves were established.
        """
        # Set up an socket stream listener
        self.__bind_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Make port reusable
        self.__bind_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # using Non-blocking IO
        self.__bind_listener.setblocking(False)
        # bind address
        self.__bind_listener.bind((self.__server_addr, self.__id_register.working_port))
        # start listening
        self.__bind_listener.listen(40)
        # reset register
        self.__id_register.reset()
        # temporary register
        _tmp_register_ref_table = [self.__bind_listener]
        _tmp_buffer_recv = {self.__bind_listener: BufferReader()}

        # wait for job submission
        while not self.__id_register.check():
            readable, writable, exp = select.select(_tmp_register_ref_table, [], _tmp_register_ref_table)
            for io_event in readable:
                if io_event is self.__bind_listener:
                    # accept connection, could be submitter or other slaves
                    con, client_address = io_event.accept()
                    # add to identified list
                    con.setblocking(False)
                    _tmp_register_ref_table.append(con)

                else:
                    buf = _tmp_buffer_recv.get(io_event, BufferReader())
                    buf.recv(io_event)
                    _tmp_buffer_recv[io_event] = buf

                    if buf.is_done():
                        data = buf.get_content()
                        # message from submitter
                        if data[Key.Type] == Type_Val.Submission:
                            self.__id_register.register(data[Key.To], data[Key.Content], io_event)
                        # message from other worker
                        elif data[Key.Type] == Type_Val.WorkerReports:
                            self.__id_register.identify(data[Key.From], data[Key.Content], io_event)

        for buf in _tmp_buffer_recv.values():
            buf.close()

        return self.__id_register

    def close(self):
        self.__bind_listener.close()


class Communication_Controller(ICommunication_Controller):

    def __init__(self, com: AbsCommunicationProcess):
        """
            Prepare communication module for connection.
            Change CommunicationController.static_server_address and CommunicationController.static_server_port
            before initializing this class.
        """
        super().__init__()
        self.__com = com
        self.__get_queue_buffer = {}

    def __enter__(self) -> ICommunication_Controller:
        self.establish_communication()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        return "Communication process, self: ({}) remote: ({}).".format(self.Node_Id, self.available_clients)

    @property
    def Com(self):
        return self.__com

    @property
    def Node_Id(self):
        return self.__com.node_id

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
            raise ConnectionAbortedError('Connection has already been closed.')
        if self.__com.recv_que.empty() and not blocking and not self.is_closed():
            return None, None
        while not self.is_closed():
            try:
                return self.__com.recv_que.get(timeout=1)
            except queue.Empty:
                continue
        raise ConnectionAbortedError('Connection is closed.')

    def send_one(self, target, dic):
        """
            send one json like object to target nodes
        :param target: target node list, must be a list : list[int]
        :param dic: json like object : encode
        :return: None
        """
        if self.is_closed():
            raise ConnectionAbortedError('Connection has already been closed.')
        if isinstance(target, list):
            self.__com.send_que.put((target, dic))
        else:
            self.__com.send_que.put(([target], dic))

        return None

    @property
    def available_clients(self):
        return self.__com.nodes

    @property
    def available_clients_count(self):
        return self.__com.available_nodes

    def close(self):
        """
            Stop communicating with remote nodes.
        :return: None
        """
        self.__com.closing()
        wait_limit = 20
        while not self.is_closed() and wait_limit > 0:
            sleep(1)
            wait_limit -= 1
        if wait_limit <= 0:
            self.__com.terminate()
            print('Terminate communication process.')

    def is_closed(self):
        """
            Check if the communication thread is already closed.
        :return: True if closed, False if still running.
        """
        return not self.__com.Alive


def get_repr():
    dns, hosts, addrs = socket.gethostbyname_ex(socket.gethostname())
    for addr in addrs:
        if addr not in {"127.0.0.1", "127.0.1.1"}:
            return addr


if __name__ == "__main__":
    from threading import Lock

    a = Lock()
    a.acquire()
