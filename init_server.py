import socketserver

from settings import GlobalSettings

from server_util.init_model import ServerUtil
from server_util.para_server import ParameterServer

from server_util.fake_com import FakeCom
from server_util.client_handler import ClientHandler


if __name__ == '__main__':
    GlobalSettings.setDefault(2, 1, 64)
    ServerUtil.initWeights()

    server = socketserver.ThreadingTCPServer(("", 15387), ClientHandler)
    para_server = ParameterServer(FakeCom())
    # register pa server
    ClientHandler.PA_Server = para_server

    server.serve_forever()
    # ServerUtil.getWeightsInit()