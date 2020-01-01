from log import Logger

from codec.pacodec import PAServerCodec
from psgd.asgd import AsynchronizedSGD
from psgd.transfer import NTransfer


class ParameterServer:

    def __init__(self, com, model, pa_codec_type=PAServerCodec):

        self.Com = com
        self.Node_ID = self.Com.Node_ID
        self.Log = Logger('ParaServer', False)
        self.Log.log_message('Init parameter server with codec {}'.format(pa_codec_type.__name__))

        updater = [{} for i in model.getWeightsInit()]

        for layer_id in range(len(updater)):
            for t in ['w', 'b']:
                updater[layer_id][t] = AsynchronizedSGD(self.Node_ID, layer_id, pa_codec_type)

        self.Transfer = NTransfer(updater, self.Com)
        self.Transfer.start_transfer()

    def post(self, sender, content):
        self.Com.SendQue.put((sender, content))
