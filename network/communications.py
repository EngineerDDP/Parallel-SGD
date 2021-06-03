import queue
import select
import socket
from time import sleep
from typing import Optional, Tuple

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

    def build_communication(self):
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
        self.__is_started = False

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
        self.__is_started = True

    def get_one(self, blocking=True, timeout: int = None) -> Tuple[Optional[int], object]:
        time_count = 0
        while time_count != timeout:
            time_count += 1
            try:
                return self.__com.get(blocking=blocking, timeout=1)
            except queue.Empty:
                if self.is_closed():
                    raise ConnectionAbortedError("Connection has already been closed, and no data available.")
        return None, None

    def send_one(self, target: [int, list], obj: object, timeout: int = None) -> bool:
        if not isinstance(target, list):
            target = [target]
        try:
            return self.__com.put(target, obj, blocking=True, timeout=timeout)
        except queue.Full:
            if self.is_closed():
                raise ConnectionAbortedError('Connection has already been closed.')
            else:
                return False

    @property
    def available_clients(self):
        return self.__com.nodes

    @property
    def available_clients_count(self):
        return self.__com.available_nodes

    def close(self, force: bool = False, timeout: int = 20):
        """
            Stop communicating with remote nodes.
        """
        if not self.__is_started:
            raise AssertionError("Start this process before closing it.")

        if force:
            self.__com.force_quit()
        else:
            self.__com.flush_data_and_quit()

        wait_limit = timeout
        while not self.is_closed() and wait_limit > 0:
            sleep(1)
            wait_limit -= 1
        if wait_limit <= 0:
            self.__com.force_quit()

    def is_closed(self):
        """
            Check if the communication thread is already closed.
        :return: True if closed, False if still running.
        """
        return self.__com.has_quit()


def get_repr():
    dns, hosts, addrs = socket.gethostbyname_ex(socket.gethostname())
    for addr in addrs:
        if addr not in {"127.0.0.1", "127.0.1.1"}:
            return addr

# ----------------------------------------
# Fix 2020年10月21日
# send_one 方法中无有效的阻断机制
# 现添加阻断机制和循环判断，当send_que 满时能够正确检查 connection 的状态并给出错误信息。
# BUG表现，大量发送的时候，send_que 很快灌满，且阻塞，无法判断连接是否失效，阻塞无法退出，
# 且不在托管状态无法使用断点调试。
# ----------------------------------------
