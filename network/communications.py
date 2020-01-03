import socket as sc

from threading import Thread

from multiprocessing import Process
from multiprocessing import Queue, Value

from time import sleep
from network.serialization import Serialize
from network.agreements import General, Initialize


class TLVPack:
    Block_Size = 1024 * 1024
    TLV_Type_Normal = 1
    TLV_Type_Exit = 0

    def __init__(self, content):
        self.Content = content
        self.Length = len(content)

    def send(self, io):
        tlv_package = TLVPack.TLV_Type_Normal.to_bytes(1, 'big') + self.Length.to_bytes(4, 'big') + self.Content
        io.sendall(tlv_package)

    @staticmethod
    def flush_garbage(io):
        b = int(0).to_bytes(4, 'big')
        io.sendall(b)

    @staticmethod
    def recv(io):
        type_ = io.recv(1)
        length = io.recv(4)
        type_ = int.from_bytes(type_, 'big')
        length = int.from_bytes(length, 'big')

        if type_ == TLVPack.TLV_Type_Exit:
            raise OSError('Connection closed by remote computer.')

        content = b''
        take = 0
        while take < length:
            read_len = min(length - take, TLVPack.Block_Size)
            content += io.recv(read_len)
            take = len(content)

        return TLVPack(content)

    @staticmethod
    def request_close(io):
        io.send(TLVPack.TLV_Type_Exit.to_bytes(1, 'big'))
        TLVPack.flush_garbage(io)


class Com(Process):
    """
        Operated with dictionary, serialized using numpy save
    """

    Circle_interval = 0.001

    def __init__(self, socketcon, nodeid=-1):
        """
            Initialize a communication control process.
        :param socketcon: socket connection object to remote device.
        :param nodeid: node id that has been assigned to this node.
        """

        Process.__init__(self, name='Communication thread address: {}'.format(socketcon.getsockname()))

        self.Connection = socketcon
        self.Node_ID = nodeid

        self.send_que = Queue()
        self.recv_que = Queue()

        self.send_thread = None
        self.recv_thread = None

        self.Exit = Value('i', 0)

    def run(self):
        """
            Bootstrap method
            start both sending and receiving thread on target socket object
        """
        self.send_thread = Thread(name='communication send.', target=self.run_send)
        self.recv_thread = Thread(name='communication recv.', target=self.run_recv)

        self.send_thread.start()
        self.recv_thread.start()

        self.Connection.settimeout(1)

        while not self.Exit.value:
            sleep(1)

        self.send_que.put(None)

        TLVPack.request_close(self.Connection)
        self.Connection.close()

        self.send_thread.join()
        self.recv_thread.join()

        self.send_que.close()
        self.recv_que.close()

        print('Communication process exited.')

    def run_send(self):
        """
            Sending thread function.
        """
        try:
            while not self.Exit.value:
                target, dic = self.send_que.get()

                if len(target) == 0:
                    continue
                # write sender info
                dic[General.From] = self.Node_ID
                # write target info
                dic[General.To] = target
                # write in TLV
                data = Serialize.pack(dic)
                pack = TLVPack(data)
                pack.send(self.Connection)
        except TypeError as e:
            pass

    def run_recv(self):
        """
            Receiving thread function.
        """
        try:
            while not self.Exit.value:
                try:
                    pack = TLVPack.recv(self.Connection)
                except sc.timeout:
                    continue
                if len(pack.Content) != 0:
                    # decode data
                    dic = Serialize.unpack(pack.Content)
                    # retrieve sender info
                    sender = dic[General.From]
                    # invoke listener
                    self.recv_que.put_nowait((sender, dic))
                pack.Content = None

        except OSError as e:
            pass
        finally:
            self.recv_que.put(None)

    def close(self):
        self.Exit.value = True


class ComConstructor:
    """
        Communication constructor
        Factory class for build class Com
    """

    def __init__(self, server, port):
        """
            Typo server address
        :param server:
        """
        self.Server = server
        self.Port = port

    def buildCom(self):
        """
            Ask the server to get a valid node id, and return a available communication
        :return: class Com, that can be bind with a specified computation node
        """
        ss = sc.socket(sc.AF_INET, sc.SOCK_STREAM)
        ss.connect((self.Server, self.Port))
        # build and init communication module
        # init dictionary
        init_dic = {General.Type: Initialize.Type}
        # pack the data
        data = Serialize.pack(init_dic)
        # pack with TLV
        pack = TLVPack(data)
        pack.send(ss)
        # retrieve data from server
        pack_return = TLVPack.recv(ss)
        dic = Serialize.unpack(pack_return.Content)

        # get node id
        node_id = dic[Initialize.Node_ID]
        # construct communication module
        com = Com(ss, node_id)
        return com


class CommunicationController:
    static_server_address = '192.168.1.140'
    static_server_port = 15387

    def __init__(self):
        """
            Prepare communication module for connection.
            Change CommunicationController.static_server_address and CommunicationController.static_server_port
            before initializing this class.
        """
        self.com_builder = ComConstructor(server=CommunicationController.static_server_address,
                                          port=CommunicationController.static_server_port)
        self.com = None
        self.Node_ID = -1

    def establish_communication(self):
        """
            Establish connection.
        :return: None
        """
        self.com = self.com_builder.buildCom()
        self.Node_ID = self.com.Node_ID
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
        :param dic: json like object : dict
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
    con = CommunicationController()
    con.establish_communication()

    print(con.Node_ID)

    con.send_one([-1], {General.Type: Initialize.Init_Weight})
    sleep(1)
    r = con.get_one()
    print(r[1])

    sleep(1)
    con.close()
