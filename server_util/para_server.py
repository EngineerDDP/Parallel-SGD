from log import Logger

from server_util.init_model import ModelMNIST

from psgd.asgd import AsynchronizedSGD
from psgd.transfer import NTransfer


class ParameterServer:

    def __init__(self, com, pa_codec):

        self.Com = com
        self.Node_ID = self.Com.Node_ID
        self.Log = Logger('*PA*', False)

        updater = [{} for i in ModelMNIST.getWeightsInit()]

        for layer_id in range(len(updater)):
            for t in ['w', 'b']:
                updater[layer_id][t] = AsynchronizedSGD(self.Node_ID, layer_id, pa_codec)

        self.Transfer = NTransfer(updater, self.Com)
        self.Transfer.start_transfer()

    def post(self, sender, content):
        self.Com.SendQue.put((sender, content))
