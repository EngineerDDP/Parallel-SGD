from log import Logger

from server_util.init_model import ServerUtil

from psgd.asgd import AsynchronizedSGD
from psgd.transfer import NTransfer

from codec.pacodec import PAServerCodec
from test.pa_codec_test import Test_PAServer


class ParameterServer:

    def __init__(self, com):

        self.Com = com
        self.Node_ID = self.Com.Node_ID
        self.Log = Logger('*PA*', False)

        updater = [{} for i in ServerUtil.getWeightsInit()]

        for layer_id in range(len(updater)):
            for t in ['w', 'b']:
                updater[layer_id][t] = AsynchronizedSGD(self.Node_ID, layer_id, Test_PAServer)

        self.Transfer = NTransfer(updater, self.Com)
        self.Transfer.start_transfer()

    def post(self, sender, content):
        self.Com.SendQue.put((sender, content))
