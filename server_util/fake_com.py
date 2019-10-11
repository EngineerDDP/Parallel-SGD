from queue import Queue

from server_util.client_handler import ClientHandler
from server_util.client_handler import Serialize

from network.agreements import DefaultNodes
from network.agreements import General
from network.communications import TLVPack, CommunicationController


class FakeCom(CommunicationController):

    def __init__(self):

        self.Node_ID = DefaultNodes.Parameter_Server
        self.SendQue = Queue()

    def is_closed(self):
        """
            PA never close.
        """
        return False

    def get_one(self):
        return self.SendQue.get()

    def send_one(self, targets, dic):

        if len(targets) == 0:
            return

        # write sender info
        dic[General.From] = self.Node_ID
        # write target info
        dic[General.To] = targets
        # write in TLV
        data = Serialize.pack(dic)
        pack = TLVPack(data)
        for target in targets:
            try:
                sock = ClientHandler.Client_List.getClient(target)
                pack.send(sock)
            except ConnectionAbortedError:
                pass

    def close(self):

        return None

    def is_closed(self):

        return False