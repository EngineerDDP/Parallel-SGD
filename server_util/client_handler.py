import socketserver
import gc
from threading import Thread

from threading import Lock

from network.agreements import General, Initialize, Transfer, DefaultNodes, Data
from network.serialization import Serialize
from settings import GlobalSettings

from server_util.init_model import ServerUtil

class DoAsync(Thread):

    def __init__(self, function, params):
        Thread.__init__(self, name="Send deamon")

        self.Function = function
        self.Params = params

    def run(self):
        self.Function(*self.Params)


class TLVPack:
    Block_Size = 1024 * 1024

    def __init__(self, content, type=1):
        self.Content = content
        self.Type = type
        self.Length = len(content)

    def send(self, io):
        tlv_package = self.Type.to_bytes(1, 'big') + self.Length.to_bytes(4, 'big') + self.Content
        i = 0
        io.sendall(tlv_package)

    def flush_garbage(io):
        b = int(0).to_bytes(100, 'big')
        io.sendall(b)

    def recv(io):
        type_ = io.recv(1)
        length = io.recv(4)
        type_ = int.from_bytes(type_, 'big')
        length = int.from_bytes(length, 'big')

        content = b''
        take = 0
        while take < length:
            read_len = min(length - take, TLVPack.Block_Size)
            content += io.recv(read_len)
            take = len(content)

        return TLVPack(content, type_)


class Node:

    def __init__(self, node_id, ip, connection):
        self.Node_ID = node_id
        self.IP = ip
        self.Connection = connection
        self.PrepareState = Initialize.State_Hold
        self.ConsistencyLock = Lock()

    def sendall(self, data):
        self.ConsistencyLock.acquire()
        self.Connection.sendall(data)
        self.ConsistencyLock.release()

    def recv(self, len):
        # self.ConsistencyLock.acquire()
        data = self.Connection.recv(len)
        # self.ConsistencyLock.release()
        return data


class NodeClients:

    def __init__(self):

        self.Clients = []
        self.Last_Node_ID = -1

    def assignNode(self, ip, connection):

        nodeid = self.Last_Node_ID + 1
        self.Clients.append(Node(nodeid, ip, connection))

        self.Last_Node_ID = nodeid

        return nodeid

    def removeNode(self, nodeid):

        i = 0
        while i < len(self.Clients):
            if self.Clients[i].Node_ID == nodeid:
                self.Clients.pop(i)
            i += 1
        if len(self.Clients) == 0:
            self.Last_Node_ID = -1

    def getClient(self, nodeid):

        for client in self.Clients:

            if client.Node_ID == nodeid:
                return client

        return None

    def isGreen(self):

        if len(self.Clients) != GlobalSettings.getDefault().NodeCount:
            return Initialize.State_Hold

        for client in self.Clients:

            if client.PrepareState == Initialize.State_Hold:
                return Initialize.State_Hold

        return Initialize.State_OK

    def setGreen(self, nodeid):

        for client in self.Clients:

            if client.Node_ID == nodeid:
                client.PrepareState = Initialize.State_OK


class ClientHandler(socketserver.BaseRequestHandler):
    GC_PER_ITERS = 100

    Client_List = NodeClients()
    PA_Server = None

    def handle(self):

        print('Connection recevied.')

        try:
            Node_ID = None
            request = self.request
            iters = 0

            while True:
                data = TLVPack.recv(request)
                if len(data.Content) == 0:
                    continue

                # record iterations
                iters += 1

                dic = Serialize.unpack(data.Content)
                if dic[General.Type] != Transfer.Type:
                    # Check for requests
                    if dic[General.Type] == Initialize.Type:
                        nodeid = ClientHandler.Client_List.assignNode(self.client_address, request)
                        dic_back = {General.Type: Initialize.Return,
                                    General.From: (-1),
                                    General.To: (-1),
                                    Initialize.Node_ID: nodeid}
                        Node_ID = nodeid
                        # request = ClientHandler.Client_List.getClient(nodeid)
                        print('Node assigned: {}'.format(nodeid))

                    elif dic[General.Type] == Initialize.Init_Weight:
                        dic_back = {General.Type: Initialize.Init_Weight,
                                    General.From: (-1),
                                    General.To: (-1),
                                    Initialize.Weight_Content: ServerUtil.getWeightsInit(),
                                    Initialize.Redundancy: GlobalSettings.getDefault().Redundancy,
                                    Initialize.Nodes: GlobalSettings.getDefault().NodeCount,
                                    Initialize.Batch_Size: GlobalSettings.getDefault().Batch.Batch_Size,
                                    Initialize.CodeType: ServerUtil.codec_ctrl(),
                                    Initialize.SyncClass: ServerUtil.psgd_type(),
                                    Initialize.Epoches: ServerUtil.epoches(),
                                    Initialize.LOSS:ServerUtil.loss_type(),
                                    Initialize.Learn_Rate:ServerUtil.learn_rate()
                                    }
                        print('Weights assigned: {}'.format(dic_back.keys()))
                    elif dic[General.Type] == Data.Type:
                        dic_back = {
                            General.Type: Initialize.Init_Weight,
                            General.From: (-1),
                            General.To: (-1),
                            Data.Train_Data: ServerUtil.train_data(),
                            Data.Eval_Data: ServerUtil.eval_data()
                        }
                        print('Data loaded: {}'.format(dic_back.keys()))
                    elif dic[General.Type] == Initialize.Current_State:
                        state = ClientHandler.Client_List.isGreen()
                        self.state_ = {General.Type: Initialize.Current_State,
                                       General.From: (-1),
                                       General.To: (-1),
                                       Initialize.Current_State: state
                                       }
                        dic_back = self.state_
                        ClientHandler.Client_List.setGreen(Node_ID)
                        # print('States check, node: {}'.format(Node_ID))
                    # Write back
                    content = Serialize.pack(dic_back)
                    tlv = TLVPack(content)
                    tlv.send(request)

                else:
                    targets = dic[General.To]

                    for target in targets:
                        # Send to PA first
                        if target == DefaultNodes.Parameter_Server:
                            ClientHandler.PA_Server.post(dic[General.From], dic)
                        else:
                            con = ClientHandler.Client_List.getClient(target)
                            if con is not None:
                                task = DoAsync(data.send, (con,))
                                task.start()
                            else:
                                print('Transfer failed, node {} not found.'.format(target))

                if iters > ClientHandler.GC_PER_ITERS:
                    gc.collect()

        except Exception as e:
            if Node_ID is not None:
                ClientHandler.Client_List.removeNode(Node_ID)
            print('Exception occurred while connecting with : {}, args : {}.'.format(self.client_address, e))
            import traceback
            traceback.print_exc()

        print('Connection aborted.')
