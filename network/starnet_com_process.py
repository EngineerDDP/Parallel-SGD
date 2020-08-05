import socket
import select
from _queue import Empty

from time import sleep

from network.agreements import *
from network.interfaces import IWorker_Register, ICommunication_Process
from network.serialization import pack, request_close, TLVPack, Serialize

STAR_NET_WORKING_PORTS = 15387


class StarNetwork_Initialization_Package:

    def __init__(self):
        self.__content = []
        self.__unique = set()

    def put(self, id, uuid, address):
        if id not in self.__unique:
            self.__content.append((str(id), uuid, address))
        else:
            raise LookupError('Id for new nodes were used before.')

    def __iter__(self):
        return self.__content.__iter__()


class Worker_Register_List:

    def __init__(self):
        self.__worker_id_to_cons = {}
        self.__fd_to_workers = {}

    def occupy(self, id: str, uuid: str):
        """
            Occupy a seat for future connections
        :param id: str
        :param uuid: str
        :return: None
        """
        self.__worker_id_to_cons[id] = uuid

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

    def identify(self, id: str, uuid: str, con: socket.socket):
        """
            identify and register a worker
        :return: bool for the result
        """
        # connect anyway if not found
        if uuid == self.__worker_id_to_cons.get(id, uuid):
            self.put(id, con)
            return True
        return False

    def put(self, id: str, con: socket.socket):
        """
            register a connection
        :param id: worker id, str
        :param con: fd
        :return: None
        """
        self.__worker_id_to_cons[id] = con
        self.__fd_to_workers[con] = id

    def rm(self, id: [str, None], con: [socket.socket, None]):
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

    def find(self, id:[str, socket.socket]):
        """
            find a connection file descriptor
        :param id: string id, to search for specified fd.
                    socket fd, to search for specified worker id.
        :return: search result
        """
        res = None
        if isinstance(id, str):
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

    def register(self, id_self, content_package:StarNetwork_Initialization_Package):
        """
            Register all workers
        :param id_self: id of current worker
        :param content_package: content package that contains address and uuid of all workers
        :return: None
        """
        self.__id = id_self

        flag = False

        # for all slaves
        for id, uuid, ip_addr in content_package:
            # slaves who's id before self
            if (not flag and id != self.__id):
                self.__workers.occupy(id, uuid)
            # id of myself
            elif (id == self.__id):
                flag = True
            # id behind myself
            elif flag:
                # try reach
                worker_con = socket.socket(socket.AF_INET, socket.SOCK_STREAM | socket.SOCK_NONBLOCK)
                try:
                    worker_con.connect((ip_addr, STAR_NET_WORKING_PORTS))
                    # try register
                    data = {
                        Key.Type: Type_Val.WorkerReports,
                        Key.From: self.__id,
                        Key.To:   id,
                        Key.Content:uuid
                    }
                    pack(data, worker_con)
                    self.__workers.put(id, worker_con)
                except OSError as error:
                    raise OSError('Error: {}, address: {}'.format(error, ip_addr))

    def put(self, id, con):
        self.__workers.put(id, con)

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
        # local thread queue
        self.__dispatch_queue = {}

    def __report_connection_lost(self, id, address):
        print('Connection with worker (id: {}, address: {}) has been lost.'.format(id, address))

    def run(self):
        """
            Bootstrap method
            start both sending and receiving thread on target socket object
        """
        import queue, threading

        for id in self.__connections.ids():
            self.__dispatch_queue[id] = queue.Queue()

        __deque_thread = threading.Thread(target=self.__run_deque, name='Communication process -> deque thread.', daemon=True)
        __deque_thread.start()

        while not self.Exit.value:
            active_connections = self.__connections.to_list()

            if len(active_connections) == 0:
                self.Exit.value = True
                break

            readable, writeable, excepts = select.select(active_connections, [], active_connections, 1)
            # read
            for fd in readable:
                try:
                    raw_pack = TLVPack.recv(fd).Content
                    data = Serialize.unpack(raw_pack)
                    _from = data[Key.From]
                    self.recv_que.put((_from, data[Key.Content]))
                except OSError as error:
                    self.__report_connection_lost(self.__connections.find(fd), fd.getpeername())
                    self.__connections.remove(fd)

            # write
            for fd in writeable:
                id = self.__connections.find(fd)
                queue = self.__dispatch_queue.get(id, None)
                if queue is not None and not queue.empty():
                    item = queue.get_nowait()
                    item.send(fd)

            # handle exception
            for fd in excepts:
                self.__report_connection_lost(self.__connections.find(fd), fd.raddr)
                self.__connections.remove(fd)
                # del queue
                id = self.__connections.find(fd)
                if self.__dispatch_queue.get(id, None) is not None:
                    del self.__dispatch_queue[id]
                    self.__dispatch_queue[id] = None

            sleep(ICommunication_Process.Circle_interval)

        for fd in self.__connections.to_list():
            request_close(fd)
            fd.close()

        self.__dispatch_queue.clear()
        self.recv_que.close()
        self.send_que.close()

        print('Communication process exited.')

    def __run_deque(self):
        """
            Sending thread function.
        """
        try:
            while not self.Exit.value:
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
                            pkg = TLVPack(Serialize.pack(pkg))
                        fd: socket.socket
                        fd.sendall()
                        pkg.send(fd)

                del data
                del pkg

        except TypeError as e:
            pass
        except OSError as e:
            pass

    def node_id(self):
        return self.__connections.get_id()

    def nodes(self):
        return self.__connections.ids()


def start_star_net(nodes: StarNetwork_Initialization_Package) -> ICommunication_Process:
    from utils.constants import Initialization_Server

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
            # serialize
            raw_pack = TLVPack(Serialize.pack(data.copy()))
            # send
            raw_pack.send(con)
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



