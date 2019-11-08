import socketserver

from settings import GlobalSettings

from server_util.init_model import ModelMNIST
from server_util.para_server import ParameterServer

from server_util.fake_com import FakeCom
from codec.test_codec.pa_codec_test import Test_PAServer
from server_util.client_handler import ClientHandler


if __name__ == '__main__':
    GlobalSettings.setDefault(2, 1, 360)
    ModelMNIST.initWeights()

    server = socketserver.ThreadingTCPServer(("", 15387), ClientHandler)
    para_server = ParameterServer(FakeCom(), Test_PAServer)
    # register pa server
    ClientHandler.PA_Server = para_server
    print('Server States GOOD.')
    print('Start listening...')
    server.serve_forever()
    # ServerUtil.getWeightsInit()