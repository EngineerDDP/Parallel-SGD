from log import Logger

from codec.paraserver import GradDiffParaServerCodec, ParaServerCodec
from codec.dc_asgdcodec import DCASGDServerCodec
from codec.sgq import SGQServer
from psgd.asgd import AsynchronizedSGD
from psgd.transfer import NTransfer


class ParameterServer:

    __para_server_map = {'simple': ParaServerCodec,
                         'graddiff': GradDiffParaServerCodec,
                         'dc': DCASGDServerCodec,
                         'sgq': SGQServer}

    def __init__(self, com, model, pa_codec_type='simple'):

        self.Com = com
        self.PA_Codec = ParameterServer.__para_server_map[pa_codec_type]
        self.Node_ID = self.Com.Node_ID
        self.Log = Logger('ParaServer', log_to_file=False)
        self.Log.log_message('Init parameter server with codec {}'.format(self.PA_Codec.__name__))

        updater = [{} for i in model.getWeightsInit()]

        for layer_id in range(len(updater)):
            for t in ['w', 'b']:
                updater[layer_id][t] = AsynchronizedSGD(self.Node_ID, layer_id, self.PA_Codec)

        self.Transfer = NTransfer(updater, self.Com)
        self.Transfer.start_transfer()

    def post(self, sender, content):
        self.Com.SendQue.put((sender, content))
