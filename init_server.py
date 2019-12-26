import socketserver

from settings import GlobalSettings

from server_util.init_model import ModelMNIST
from server_util.para_server import ParameterServer

from server_util.fake_com import FakeCom
from codec.test_codec.pa_codec_test import Test_PAServer
from codec.pacodec import PAServerCodec
from server_util.client_handler import ClientHandler

import sys


if __name__ == '__main__':

    Node_count = 1
    Batch_size = 128
    Redundancy = 1

    if len(sys.argv) > 1:
        Node_count = int(sys.argv[1])
    if len(sys.argv) > 2:
        Batch_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        Redundancy = int(sys.argv[3])

    GlobalSettings.setDefault(Node_count, Redundancy, Batch_size)
    ModelMNIST.initWeights()

    port = 15387

    server = socketserver.ThreadingTCPServer(("", port), ClientHandler)
    para_server = ParameterServer(FakeCom(), PAServerCodec)
    # register pa server
    ClientHandler.PA_Server = para_server
    ClientHandler.CallBack_Server = server
    print('Server States GOOD.')
    print('Start listening: 0.0.0.0:{} '.format(port))
    server.serve_forever()
    # ServerUtil.getWeightsInit()