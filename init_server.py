from profiles.settings import GlobalSettings
from server_util.init_model import ModelDNN
from server_util.para_server import ParameterServer
from server_util.fake_com import FakeCom
from server_util.client_handler import ServerParameters, start_server, create_logger
from dataset.mnist_input import load_mnist
from profiles.blockassignment.idependent import IIDBlockAssignment

import argparse


if __name__ == '__main__':

    """
        Usage:
        python init_server.py --node_count 4 --batch_size 128 --redundancy 2 --codec ccdc \
         --psgd ssgd --learn_rate 0.05 --epochs 10 --working_ports 15387 --block_assignment iid \
         --server_codec graddiff
    """
    parse = argparse.ArgumentParser()
    parse.add_argument("--node_count", type=int, default=3, help="initial node count")
    parse.add_argument("--batch_size", type=int, default=192, help="initial batch size")
    parse.add_argument("--redundancy", type=int, default=1, help="initial redundancy")
    parse.add_argument("--codec", type=str, default='sgq', help="initial communication codec and protocol {ccdc, plain, psclient}")
    parse.add_argument("--psgd", type=str, default='ssgd', help="parallel stochastic gradient descent synchronization type {asgd, ssgd}")
    parse.add_argument("--learn_rate", type=float, default=0.05, help="initial learining rate")
    parse.add_argument("--epochs", type=int, default=100, help="initial train epochs")
    parse.add_argument("--working_ports", type=int, default=55555, help="initial working ports")
    parse.add_argument("--block_assignment", type=str, default='iid', help="initial block assignment strategy")
    parse.add_argument("--server_codec", type=str, default='sgq', help="server codec for parameter averaging")
    arg = parse.parse_args()

    # read parameters
    node_count = arg.node_count
    batch_size = arg.batch_size
    redundancy = arg.redundancy
    codec = arg.codec
    psgd = arg.psgd
    lr = arg.learn_rate
    epo = arg.epochs
    port = arg.working_ports
    assignment = arg.block_assignment
    server_codec = arg.server_codec

    Global_Logger = create_logger(port)

    # For debug propose
    if port == 55555:
        Global_Logger.ToFile = False

    Global_Logger.log_message('\t --node_count <node count {}>'.format(node_count))
    Global_Logger.log_message('\t --batch_size <batch size {}>'.format(batch_size))
    Global_Logger.log_message('\t --redundancy <r {}>'.format(redundancy))
    Global_Logger.log_message('\t --codec <communication codec and protocol {}>'.format(codec))
    Global_Logger.log_message('\t --psgd <parallel stochastic gradient descent synchronization type {}>'.format(psgd))
    Global_Logger.log_message('\t --learn_rate <learn rate for GD algorithm {}>'.format(lr))
    Global_Logger.log_message('\t --epochs <training epochs {}>'.format(epo))
    Global_Logger.log_message('\t --working_ports <ports activated {}>'.format(port))
    Global_Logger.log_message('\t --block_assignment <block assignment strategy {}>'.format(assignment))
    Global_Logger.log_message('\t --server_codec <parameter server codec {}>'.format(server_codec))

    train_x, train_y = load_mnist(kind='train')
    test_x, test_y = load_mnist(kind='t10k')

    # Set global settings
    GlobalSettings.set_default(node_count, redundancy, batch_size, IIDBlockAssignment)
    # Set model parameters
    model_parameter = ModelDNN(train_x=train_x, train_y=train_y,
                               test_x=test_x, test_y=test_y,
                               codec=codec,
                               psgd_type=psgd,
                               learn_rate=lr,
                               epoches=epo,
                               block_assignment=assignment)
    # Set parameter server
    para_server = ParameterServer(FakeCom(), model_parameter, pa_codec_type=server_codec)

    # Register settings
    ServerParameters.set_default(model_parameter, para_server)
    start_server(port)
