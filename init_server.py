from settings import GlobalSettings
from server_util.init_model import ModelDNN
from server_util.para_server import ParameterServer
from server_util.fake_com import FakeCom
from server_util.client_handler import ServerParameters, start_server
from dataset.mnist_input import load_mnist

import sys


if __name__ == '__main__':

    """
        Usage:
        python init_server.py --node_count 4 --batch_size 128 --redundancy 2 --codec ccdc \
         --psgd ssgd --learn_rate 0.05 --epochs 10 --working_ports 15387
    """

    if len(sys.argv) < 15:
        print('usage: init_server.py')
        print('\t --node_count <node count>')
        print('\t --batch_size <batch size>')
        print('\t --redundancy <r>')
        print('\t --codec <communication codec and protocol> {ccdc, plain, psclient}')
        print('\t --psgd <parallel stochastic gradient descent synchronization type> {asgd, ssgd}')
        print('\t --learn_rate <learn rate for GD algorithm>')
        print('\t --epochs <training epochs>')
        print('\t --working_ports <ports activated>')
        exit(-1)

    # read parameters
    node_count = int(sys.argv[2])
    batch_size = int(sys.argv[4])
    redundancy = int(sys.argv[6])
    codec = sys.argv[8]
    psgd = sys.argv[10]
    lr = float(sys.argv[12])
    epo = int(sys.argv[14])
    port = int(sys.argv[16])

    train_x, train_y = load_mnist(kind='train')
    test_x, test_y = load_mnist(kind='t10k')

    # Set global settings
    GlobalSettings.set_default(node_count, redundancy, batch_size)
    # Set model parameters
    model_parameter = ModelDNN(train_x=train_x, train_y=train_y,
                               test_x=test_x, test_y=test_y,
                               codec=codec,
                               psgd_type=psgd,
                               learn_rate=lr,
                               epoches=epo)
    # Set parameter server
    para_server = ParameterServer(FakeCom(), model_parameter)

    # Register settings
    ServerParameters.set_default(model_parameter, para_server)
    start_server(port)
