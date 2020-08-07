import socket
import select
from multiprocessing import Array, Value
from queue import Empty

from time import sleep

from network.agreements import *
from network.interfaces import IWorker_Register, ICommunication_Process
from network.serialization import Buffer

from utils.constants import Initialization_Server
STAR_NET_WORKING_PORTS = 15387


class StarNetwork_Initialization_Package:

    def __init__(self):
        self.__content = []
        self.__unique = set()

    def put(self, id, uuid, address):
        if id not in self.__unique:
            self.__content.append((id, uuid, address))
        else:
            raise LookupError('Id for new nodes were used before.')

    def __iter__(self):
        return self.__content.__iter__()


class Worker_Register_List:

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
                    Buffer.request_close(_con)
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

    def rm(self, id: [int, None], con: [socket.socket, None]):
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

    def find(self, id:[int, socket.socket]):
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

class Worker_Register(IWorker_Register):

    def __init__(self):
        self.__id = None
        self.__workers = Worker_Register_List()

    def __iter__(self):
        return self.__workers.to_list()

    def to_list(self):
        return self.__workers.to_list()

    def register(self, id_self, content_package:StarNetwork_Initialization_Package, con_from=None):
        """
            Register all workers
        :param id_self: id of current worker
        :param content_package: content package that contains address and uuid of all workers
        :return: None
        """
        self.__id = id_self
        if con_from is not None:
            self.__workers.put(Initialization_Server, con_from)

        self_uuid = None

        # for all slaves
        for id, uuid, ip_addr in content_package:
            # slaves who's id before self
            if (self_uuid is None and id != self.__id):
                self.__workers.occupy(id, uuid)
            # id of myself
            elif (id == self.__id):
                self_uuid = uuid
            # id behind myself
            elif self_uuid is not None:
                # try reach
                worker_con = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    worker_con.connect((ip_addr, STAR_NET_WORKING_PORTS))
                    # try register
                    data = {
                        Key.Type:       Type_Val.WorkerReports,
                        Key.From:       self.__id,
                        Key.To:         id,
                        Key.Content:    self_uuid
                    }
                    pkg = Buffer(content=data)
                    pkg.send(worker_con)
                    self.__workers.put(id, worker_con)
                    worker_con.setblocking(False)
                    pkg.close()

                except OSError as error:
                    raise OSError('Error: {}, address: {}'.format(error, ip_addr))

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


class Communication_Process(ICommunication_Process):
    """
        Operated with dictionary, serialized using numpy save
    """

    def __init__(self, id_register: Worker_Register):
        """
            Initialize a communication control process.
        :param socketcon: socket connection object to remote device.
        :param nodeid: node id that has been assigned to this node.
        """

        super().__init__(name='Communication thread address: {}'.format(id_register.get_id()))
        self.__connections = id_register
        self.__available_nodes_count = Value('i', len(self.__connections.ids()))
        self.__available_nodes = Array('i', self.__connections.ids())

        # do detach
        for fd in self.__connections.to_list():
            fd.set_inheritable(True)

    def __report_connection_lost(self, fd, address):
        self.__available_nodes_count.value -= 1
        id = self.__connections.find(fd)
        self.__connections.remove(fd)
        self.__available_nodes[:self.__available_nodes_count.value] = self.__connections.ids()
        print('Connection with worker (id: {}, address: {}) has been lost.'.format(id, address))

    def run(self):
        """
            Bootstrap method
            start both sending and receiving thread on target socket object
        """
        import threading

        recv_buffer_list = {}
        for fd in self.__connections.to_list():
            recv_buffer_list[fd] = Buffer()
            fd.setblocking(False)

        __deque_thread = threading.Thread(target=self.__run_deque, name='Communication process -> deque thread.', daemon=True)
        __deque_thread.start()

        while not self.Exit:
            active_connections = self.__connections.to_list()

            if len(active_connections) == 0:
                self.Exit = True
                break

            readable, writeable, excepts = select.select(active_connections, [], active_connections, 1)
            # read
            for fd in readable:
                try:
                    buf = recv_buffer_list[fd]
                    buf.recv(fd)
                    if buf.is_ready():
                        data = buf.get_content()
                        _from = data[Key.From]
                        self.recv_que.put((_from, data[Key.Content]))

                except OSError as error:
                    recv_buffer_list[fd].close()
                    self.__report_connection_lost(fd, fd.getpeername())
                    #
                    # # print DEBUG message
                    # import sys
                    # import traceback
                    # exc_type, exc_value, exc_tb = sys.exc_info()
                    # traceback.print_exception(exc_type, exc_value, exc_tb)
                    # # print DEBUG message

            # # write
            # for fd in writeable:
            #     id = self.__connections.find(fd)
            #     queue = self.__dispatch_queue.get(id, None)
            #     if queue is not None and not queue.empty():
            #         item = queue.get_nowait()
            #         item.send(fd)

            # handle exception
            for fd in excepts:
                recv_buffer_list[fd].close()
                self.__report_connection_lost(fd, fd.raddr)

            # sleep(ICommunication_Process.Circle_interval)

        for fd in self.__connections.to_list():
            Buffer.request_close(fd)
            fd.close()

        self.recv_que.close()
        self.send_que.close()

        print('Communication process exited.')

    def __run_deque(self):
        """
            Sending thread function.
        """
        send_buf = Buffer()
        try:
            while not self.Exit:
                try:
                    target, data = self.send_que.get(timeout=1)
                except Empty:
                    continue

                pkg = None

                for id in target:
                    fd = self.__connections.find(id)
                    if fd is not None:
                        if pkg is None:
                            pkg = {
                                Key.Type:       Type_Val.Normal,
                                Key.From:       self.__connections.get_id(),
                                Key.To:         target,
                                Key.Content:    data
                            }
                            # write in TLV
                            send_buf.set_content(pkg)

                        send_buf.send(fd)

                del data
                del pkg

        except TypeError as e:
            pass
        except OSError as e:
            pass
        finally:
            send_buf.close()

    @property
    def node_id(self):
        return self.__connections.get_id()

    @property
    def nodes(self):
        return self.__available_nodes[:self.__available_nodes_count.value]

    @property
    def available_nodes(self):
        return self.__available_nodes_count


def start_star_net(nodes: StarNetwork_Initialization_Package) -> ICommunication_Process:

    worker_register = Worker_Register()
    # register
    worker_register.register(Initialization_Server, nodes)
    data = {
        Key.Type: Type_Val.Submission,
        Key.From: Initialization_Server,
        Key.To: -1,
        Key.Content: nodes
    }

    try:
        for id, uuid, address in nodes:
            con = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # connect
            con.connect((address, STAR_NET_WORKING_PORTS))
            # write key
            data[Key.To] = id
            # serialize and send
            pkg = Buffer(data)
            pkg.send(con)
            pkg.close()
            # add
            worker_register.identify(id, uuid, con=con)

    except OSError as error:
        for con in worker_register.to_list():
            if isinstance(con, socket.socket):
                con.close()
        raise OSError('Error: {}.'.format(error))

    if worker_register.check():
        com = Communication_Process(worker_register)
        return com
    else:
        raise OSError('Some of workers didnt respond properly.')



