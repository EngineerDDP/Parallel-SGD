import queue
import select
import socket
import threading
from typing import Sequence, Tuple, Union

import constants
from network.agreements import *
from network.interfaces import INodeRegister, AbsCommunicationProcess, IPromoter, NodeAssignment
from network.serialization import BufferReader, BufferWriter

STAR_NET_WORKING_PORTS = constants.Network_Working_Ports


class WorkerRegisterList:

    def __init__(self):
        self.__worker_id_to_cons = {}
        self.__fd_to_workers = {}

    def occupy(self, id: int, uuid: str):
        """
            Occupy a seat for future connections
        :param id: str
        :param uuid: str
        :return: None
        """
        _tmp_cons = self.__worker_id_to_cons.get(id, None)
        # if not settled
        if _tmp_cons is None:
            self.__worker_id_to_cons[id] = uuid
        # check which one is right
        elif isinstance(_tmp_cons, list):
            for _uuid, _con in _tmp_cons:
                if _uuid == uuid:
                    self.put(id, _con)
                # close it
                else:
                    BufferWriter.request_close(_con)
                    _con.close()

    def check(self):
        """
            check if all the occupied seats were filled with real connections.
        :return: bool
        """
        if len(self.__worker_id_to_cons) == 0:
            return False
        for key, val in self.__worker_id_to_cons.items():
            if not isinstance(val, socket.socket):
                return False
        return True

    def identify(self, id: int, uuid: str, con: socket.socket):
        """
            identify and register a worker
        :return: bool for the result
        """
        _uuid = self.__worker_id_to_cons.get(id, None)
        # stash connection
        if uuid == _uuid:
            self.put(id, con)
        elif _uuid is None:
            self.__worker_id_to_cons[id] = [(uuid, con)]
        elif isinstance(_uuid, list):
            _uuid.append((uuid, con))
        else:
            return False

        return True

    def put(self, id: int, con: socket.socket):
        """
            register a connection
        :param id: worker id, int
        :param con: fd
        :return: None
        """
        self.__worker_id_to_cons[id] = con
        self.__fd_to_workers[con] = id

    def rm(self, id: Union[int, None], con: Union[socket.socket, None]):
        """
            remove a connection
        :param id: worker id, None or str
        :param con: connection fd, None or socket
        :return: None
        """
        if id is not None and self.find(id) is not None:
            con = self.__worker_id_to_cons[id]
            del self.__worker_id_to_cons[id]
            del self.__fd_to_workers[con]

        elif con is not None and self.find(con) is not None:
            id = self.__fd_to_workers[con]
            del self.__fd_to_workers[con]
            del self.__worker_id_to_cons[id]

    def find(self, id: Union[int, socket.socket]):
        """
            find a connection file descriptor
        :param id: integer id, to search for specified fd.
                    socket fd, to search for specified worker id.
        :return: search result
        """
        res = None
        if isinstance(id, int):
            res = self.__worker_id_to_cons.get(id, None)
        if isinstance(id, socket.socket):
            res = self.__fd_to_workers.get(id, None)
        return res

    def to_list(self):
        return list(self.__worker_id_to_cons.values())

    def keys(self):
        return list(self.__fd_to_workers.values())


class NodeRegister(INodeRegister):

    def __init__(self):
        super().__init__()
        self.__id = None
        self.__workers = WorkerRegisterList()

    def __iter__(self):
        return self.__workers.to_list()

    def to_list(self):
        return self.__workers.to_list()

    @property
    def working_port(self):
        return STAR_NET_WORKING_PORTS

    def register(self, id_self, content_package: NodeAssignment, con_from=None):
        """
            Register all workers
        :param id_self: id of current worker
        :param content_package: content package that contains address and uuid of all workers
        :return: None
        """
        self.__id = id_self
        if con_from is not None:
            self.__workers.put(constants.Promoter_ID, con_from)

        self_uuid = None
        uuid = content_package.uuid
        writer = BufferWriter()

        # for all slaves
        for id, ip_addr in content_package:
            # slaves who's id before self
            if self_uuid is None and id != self.__id:
                self.__workers.occupy(id, uuid)
            # id of myself
            elif id == self.__id:
                self_uuid = uuid
            # id behind myself
            elif self_uuid is not None:
                # try reach
                worker_con = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    worker_con.connect((ip_addr, STAR_NET_WORKING_PORTS))
                    # try register
                    data = {
                        Key.Type: Type_Val.WorkerReports,
                        Key.From: self.__id,
                        Key.To: id,
                        Key.Content: self_uuid
                    }
                    writer.set_content(data)
                    writer.send(worker_con)
                    self.__workers.put(id, worker_con)
                    worker_con.setblocking(False)

                except OSError as error:
                    raise OSError('Error: {}, address: {}'.format(error, ip_addr))

        writer.close()

    def find(self, id):
        return self.__workers.find(id)

    def logout(self, con):
        self.__workers.rm(id=None, con=con)

    def identify(self, id, content, con):
        self.__workers.identify(id, content, con)

    def check(self):
        return self.__id is not None and self.__workers.check()

    def get_id(self):
        return self.__id

    def remove(self, con):
        return self.__workers.rm(id=None, con=con)

    def ids(self):
        return self.__workers.keys()

    def reset(self):
        del self.__workers
        self.__workers = WorkerRegisterList()


class CommunicationProcess(AbsCommunicationProcess):
    """
        Operated with dictionary, serialized using numpy save
    """

    def __init__(self, id_register: NodeRegister):
        """
            Initialize a communication control process.
        :param id_register: Network registration for identifying distributed nodes.
        """

        self.__connections = id_register
        self.__available_nodes_count: int = len(self.__connections.ids())
        self.__available_nodes: Sequence[int] = self.__connections.ids()
        self.__data_bytes_sent = 0
        self.__data_bytes_received = 0
        # do detach and set Async-IO
        for fd in self.__connections.to_list():
            fd.set_inheritable(True)
            fd.setblocking(False)
        # send and recv
        self.__send_thread = threading.Thread(name="Net Send ID: {}".format(id_register.get_id()),
                                              target=self.__send_proc,
                                              daemon=True)
        self.__recv_thread = threading.Thread(name="Net Recv ID: {}".format(id_register.get_id()),
                                              target=self.__recv_proc,
                                              daemon=True)
        self.__send_queue = queue.Queue(maxsize=constants.Network_Queue_Size)
        self.__recv_queue = queue.Queue(maxsize=constants.Network_Queue_Size)
        # exit mark
        self.__exit_mark = False
        # true quit mark
        self.__quit_mark = False

    def __report_connection_lost(self, fd, address):
        """
        Report a connections lost.
        :param fd:
        :param address:
        :return: None
        """
        # get fd
        _lost_id = self.__connections.find(fd)
        self.__connections.remove(fd)
        fd.close()
        # record
        self.__available_nodes_count -= 1
        self.__available_nodes[:self.__available_nodes_count] = self.__connections.ids()
        # Skip this to comfort users.
        # print('Connection with worker (id: {}, address: {}) has been lost.'.format(id, address))

    def start(self):
        self.__send_thread.start()
        self.__recv_thread.start()

    def force_quit(self):
        # blocking put()
        self.flush_data_and_quit()
        # multi-thread safe
        while not self.__send_queue.empty():
            try:
                self.__send_queue.get_nowait()
            except queue.Empty:
                pass

    def flush_data_and_quit(self):
        self.__exit_mark = True

    def put(self, target: Union[Sequence[int], int], obj: object, blocking: bool, timeout: int):
        # valid target, safe guard
        if isinstance(target, list):
            for item in target:
                if not isinstance(item, int):
                    raise ValueError("Send targets can only be int, but {} received.".format(type(item)))  # time saving
        else:
            raise ValueError("Send targets can only be int, but {} received.".format(type(target)))  # time saving
        # do real things
        if not self.__exit_mark:
            try:
                self.__send_queue.put((target, obj), block=blocking, timeout=timeout)
                return True
            except queue.Full:
                return False
        else:
            return False

    def get(self, blocking: bool, timeout: int) -> Tuple[int, object]:
        return self.__recv_queue.get(block=blocking, timeout=timeout)

    def has_quit(self) -> bool:
        return self.__quit_mark

    def __request_close(self):
        active_connections = self.__connections.to_list()
        try:
            while len(active_connections) != 0:
                _, writable, _ = select.select([], active_connections, [], 1)
                for fd in writable:
                    active_connections.remove(fd)
                    BufferWriter.request_close(fd)
                    fd.close()
        except ConnectionResetError:
            pass

    def __recv_proc(self):
        """
            Bootstrap method
            start both sending and receiving thread on target socket object
        """

        recv_buffer_list = {}
        for fd in self.__connections.to_list():
            recv_buffer_list[fd] = BufferReader()

        # handle readable event here
        while not self.__exit_mark:
            active_connections = self.__connections.to_list()

            if len(active_connections) == 0:
                self.__exit_mark = True
                break
            try:
                readable, _, excepts = select.select(active_connections, [], active_connections, 1)
            except ValueError:
                continue
            # read
            for fd in readable:
                try:
                    buf = recv_buffer_list[fd]
                    buf.recv(fd)
                    if buf.is_done():
                        # record length
                        self.__data_bytes_received += len(buf)
                        # do decode
                        data = buf.get_content()
                        _from = data[Key.From]
                        try:
                            self.__recv_queue.put_nowait((_from, data[Key.Content]))
                        except queue.Full:
                            pass

                except OSError as error:
                    recv_buffer_list[fd].close()
                    addr = 'Unknown'
                    try:
                        addr = fd.getpeername()
                    except Exception:
                        pass
                    self.__report_connection_lost(fd, addr)
                    #
                    # # print DEBUG message
                    # import sys
                    # import traceback
                    # exc_type, exc_value, exc_tb = sys.exc_info()
                    # traceback.print_exception(exc_type, exc_value, exc_tb)
                    # # print DEBUG message

            # handle exception
            for fd in excepts:
                recv_buffer_list[fd].close()
                self.__report_connection_lost(fd, fd.raddr)

            # sleep(ICommunication_Process.Circle_interval)

        self.__send_thread.join()
        self.__request_close()

        self.__connections.reset()
        self.__quit_mark = True

    def __send_proc(self):
        """
            Sending thread function.
        """
        writing_list = {}
        active_connections = []
        tmp_item = None
        # if not exit and workable
        while not self.__exit_mark or not self.__send_queue.empty() or len(active_connections) != 0:
            # if network is idle, or queue has items
            if len(active_connections) == 0 or self.__send_queue.qsize() > 0:
                if isinstance(tmp_item, tuple):
                    target, data = tmp_item  # cond: (len(a) > 0 and buff is not valid and qsize > 0)
                    tmp_item = None
                else:
                    try:
                        target, data = self.__send_queue.get(
                            timeout=1)  # cond: (len(a) == 0 and qsize == 0) or (qsize > 0)
                    except queue.Empty:
                        continue

                left = list()
                for send_to in target:
                    fd = self.__connections.find(send_to)
                    if fd is None:
                        continue

                    # cond: (send_to is valid) and (fd is valid)
                    buffer = writing_list.get(fd, BufferWriter())
                    writing_list[fd] = buffer
                    # if not full
                    if buffer.is_done():
                        pkg = {
                            Key.Type: Type_Val.Normal,
                            Key.From: self.__connections.get_id(),
                            Key.To: target,
                            Key.Content: data
                        }
                        # cond: (buffer is valid) and (send_to is valid) and (fd is valid)
                        # do encode
                        buffer.set_content(pkg)
                        active_connections.append(fd)
                        # record length
                        self.__data_bytes_sent += len(buffer)
                    # put it back
                    else:
                        if fd not in active_connections:
                            active_connections.append(fd)
                        left.append(send_to)

                if len(left) > 0:
                    # cond: (qfull or not qfull) and tmp_item = None
                    tmp_item = (left, data)
                    if not self.__send_queue.full():
                        try:
                            self.__send_queue.put_nowait(tmp_item)
                            # cond: put valid and len(left) > 0
                            tmp_item = None
                        except queue.Full:
                            pass
            # do send jobs
            if len(active_connections) > 0:
                try:
                    _, writable, excepts = select.select([], active_connections, [], 1)
                except ValueError:
                    continue

                for fd in writable:
                    try:
                        buf = writing_list[fd]
                        buf.send(fd)
                        if buf.is_done():
                            active_connections.remove(fd)
                    # ignore exceptions and left it for main thread to handle.
                    except OSError:
                        active_connections.remove(fd)

                for fd in excepts:
                    active_connections.remove(fd)

        for buf in writing_list.values():
            buf.close()

    @property
    def node_id(self):
        return self.__connections.get_id()

    @property
    def nodes(self):
        return self.__available_nodes[:self.__available_nodes_count]

    @property
    def available_nodes(self):
        return self.__available_nodes_count

    @property
    def bytes_read(self):
        return self.__data_bytes_received

    @property
    def bytes_sent(self):
        return self.__data_bytes_sent


class Promoter(IPromoter):

    def __call__(self, nodes: NodeAssignment) -> AbsCommunicationProcess:
        worker_register = NodeRegister()
        # register
        worker_register.register(constants.Promoter_ID, nodes)
        data = {
            Key.Type: Type_Val.Submission,
            Key.From: constants.Promoter_ID,
            Key.To: -1,
            Key.Content: nodes
        }

        writer = BufferWriter()
        uuid = nodes.uuid

        for id, address in nodes:
            try:
                con = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # connect
                con.connect((address, STAR_NET_WORKING_PORTS))
                # write key
                data[Key.To] = id
                # serialize and send
                writer.set_content(data)
                writer.send(con)
                # add
                worker_register.identify(id, uuid, con=con)

            except OSError as error:
                for con in worker_register.to_list():
                    if isinstance(con, socket.socket):
                        con.close()
                raise OSError('{}, while connecting {}.'.format(error, address))

        writer.close()

        if worker_register.check():
            com = CommunicationProcess(worker_register)
            return com
        else:
            raise OSError('Some of workers didnt respond properly.')

# ----------------------------------------
# Fix 2020年10月22日
# 为主发送线程添加异常捕捉机制
# __run_deque 方法中无有效的异常处理机制
# 为 __run_deque 方法添加两个异常处理机制：
# 1. 当正在发送的发送列表失效，使用select做exception捕捉
# 2. 当正在发送的发送列表fd失效，使用try catch 捕捉 select 异常
# ----------------------------------------
# ----------------------------------------
# Fix 2020年10月25日
# 为主发送线程添加异常捕捉机制
# 用于在被动退出时防止fd失效导致的崩溃
# ----------------------------------------
