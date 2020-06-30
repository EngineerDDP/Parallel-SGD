import socketserver
import gc
from threading import Thread

from threading import Lock
from threading import Event

from network.agreements import General, Initialize, Transfer, DefaultNodes, Data
from network.serialization import Serialize
from profiles.settings import GlobalSettings

from network.communications import TLVPack

from log import Logger

Global_Logger = None

def create_logger(port):
    global Global_Logger
    Global_Logger = Logger('Server_{}'.format(port), log_to_file=True)
    Global_Logger.log_message('Starting server at port {}.'.format(port))
    return Global_Logger

def start_server(port):
    server = socketserver.ThreadingTCPServer(("", port), ClientHandler)
    server.serve_forever()


class DoAsync(Thread):

    def __init__(self, function, params):
        Thread.__init__(self, name="Send deamon")

        self.Function = function
        self.Params = params

    def run(self):
        self.Function(*self.Params)


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

        if len(self.Clients) != GlobalSettings.get_default().node_count:
            return Initialize.State_Hold

        for client in self.Clients:

            if client.PrepareState == Initialize.State_Hold:
                return Initialize.State_Hold

        return Initialize.State_OK

    def setGreen(self, nodeid):

        for client in self.Clients:

            if client.Node_ID == nodeid:
                client.PrepareState = Initialize.State_OK


class ServerParameters:

    __default = None
    __signal = Event()

    def __init__(self, model_settings, pa_server):
        self.ModelSetting = model_settings
        self.ParameterServer = pa_server

    @staticmethod
    def get_default():
        if ServerParameters.__default is None:
            ServerParameters.__signal.wait()
        return ServerParameters.__default

    @staticmethod
    def set_default(model_settings, pa_server):
        ServerParameters.__default = ServerParameters(model_settings, pa_server)


class ClientHandler(socketserver.BaseRequestHandler):
    GC_PER_ITERS = 100

    Client_List = NodeClients()

    def handle(self):

        Global_Logger.log_message('Connection recevied.')

        model = ServerParameters.get_default().ModelSetting
        pa_server = ServerParameters.get_default().ParameterServer
        node_id = None
        data_posted = 0

        try:
            request = self.request
            iters = 0

            while True:
                data = TLVPack.recv(request)
                if len(data.Content) == 0:
                    continue

                data_posted += len(data.Content)

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
                        node_id = nodeid
                        # request = ClientHandler.Client_List.getClient(nodeid)
                        Global_Logger.log_message('Node assigned: {}'.format(nodeid))

                    elif dic[General.Type] == Initialize.Init_Weight:
                        dic_back = {General.Type: Initialize.Init_Weight,
                                    General.From: (-1),
                                    General.To: (-1),
                                    Initialize.Weight_Content: model.getWeightsInit(),
                                    Initialize.Redundancy: GlobalSettings.get_default().redundancy,
                                    Initialize.Nodes: GlobalSettings.get_default().node_count,
                                    Initialize.Batch_Size: GlobalSettings.get_default().batch.batch_size,
                                    Initialize.CodeType: model.codec_ctrl(),
                                    Initialize.SyncClass: model.psgd_type(),
                                    Initialize.Epoches: model.epoches(),
                                    Initialize.LOSS:model.loss_type(),
                                    Initialize.Learn_Rate:model.learn_rate(),
                                    Initialize.Target_Accuracy:model.target_acc(),
                                    Initialize.Block_Assignment:model.Block_Assignment
                                    }
                        Global_Logger.log_message('Weights assigned: {}'.format(dic_back.keys()))
                    elif dic[General.Type] == Data.Type:
                        dic_back = {
                            General.Type: Data.Type,
                            General.From: (-1),
                            General.To: (-1),
                            Data.Train_Data: model.train_data(),
                            Data.Eval_Data: model.eval_data()
                        }
                        Global_Logger.log_message('Data loaded: {}'.format(dic_back.keys()))
                    elif dic[General.Type] == Initialize.Current_State:
                        state = ClientHandler.Client_List.isGreen()
                        state_dic  =   {General.Type: Initialize.Current_State,
                                        General.From: (-1),
                                        General.To: (-1),
                                        Initialize.Current_State: state
                                       }
                        dic_back = state_dic
                        ClientHandler.Client_List.setGreen(node_id)
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
                            pa_server.post(dic[General.From], dic)
                        else:
                            con = ClientHandler.Client_List.getClient(target)
                            if con is not None:
                                task = DoAsync(data.send, (con,))
                                task.start()
                            else:
                                Global_Logger.log_message('Transfer failed, node {} not found.'.format(target))

                if iters > ClientHandler.GC_PER_ITERS:
                    gc.collect()

        except Exception as e:
            if node_id is not None:
                ClientHandler.Client_List.removeNode(node_id)
            Global_Logger.log_error('Exception occurred while connecting with : {}, args : {}.'.format(self.client_address, e))
            import traceback
            traceback.print_exc()

        Global_Logger.log_message('Connection aborted with Node: {}, Data received: {}'.format(node_id, data_posted))
