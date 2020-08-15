import time

from profiles.settings import GlobalSettings
from server_util.init_model import ModelDNN, ModelCNN
from dataset.mnist_input import load_mnist
from coordinator import Coordinator

import argparse
import json

from utils.log import Logger

if __name__ == '__main__':

    """
        Usage:
        python job_submit.py --node_count 4 --batch_size 128 --redundancy 2 --codec ccdc \
         --psgd ssgd --learn_rate 0.05 --epochs 10 --block_assignment iid \
         --server_codec graddiff --workers worker.json
    """
    parse = argparse.ArgumentParser()
    parse.add_argument("--retrieve_data", action="store_true", dest="do_retrieve_only", default=False, help="retrieve data from cluster and exit.")
    parse.add_argument("-n", "--node_count", type=int, default=1, help="initial node count")
    parse.add_argument("-b", "--batch_size", type=int, default=64, help="initial batch size")
    parse.add_argument("-r", "--redundancy", type=int, default=1, help="initial redundancy")
    parse.add_argument("-c", "--codec", type=str, default='plain', help="initial communication codec and protocol {ccdc, plain, ps}")
    parse.add_argument("--optimizer", type=str, default='psgd', help="Optimizer used for model training.")
    parse.add_argument("--psgd", type=str, default='ssgd', help="parallel stochastic gradient descent synchronization type {asgd, ssgd}")
    parse.add_argument("--learn_rate", type=float, default=0.05, help="initial learining rate")
    parse.add_argument("--epochs", type=int, default=2, help="initial train epochs")
    parse.add_argument("--block_assignment", type=str, default='iid', help="initial block assignment strategy")
    parse.add_argument("--server_codec", type=str, default='sgq', help="server codec for parameter averaging")
    parse.add_argument("--workers", type=str, default='worker.json', help='worker list file, json type')
    arg = parse.parse_args()

    # read parameters
    node_count = arg.node_count
    batch_size = arg.batch_size
    redundancy = arg.redundancy
    codec = arg.codec
    psgd = arg.psgd
    lr = arg.learn_rate
    epo = arg.epochs
    assignment = arg.block_assignment
    server_codec = arg.server_codec
    op = arg.optimizer
    logger = Logger(title_info='User Submit', log_to_file=True)

    logger.log_message('\t --node_count <node count {}>'.format(node_count))
    logger.log_message('\t --batch_size <batch size {}>'.format(batch_size))
    logger.log_message('\t --redundancy <r {}>'.format(redundancy))
    logger.log_message('\t --codec <communication codec and protocol {}>'.format(codec))
    logger.log_message('\t --optimizer <optimizer for model training {}>'.format(op))
    logger.log_message('\t --psgd <parallel stochastic gradient descent synchronization type {}>'.format(psgd))
    logger.log_message('\t --learn_rate <learn rate for GD algorithm {}>'.format(lr))
    logger.log_message('\t --epochs <training epochs {}>'.format(epo))
    logger.log_message('\t --block_assignment <block assignment strategy {}>'.format(assignment))
    logger.log_message('\t --server_codec <parameter server codec {}>'.format(server_codec))

    train_x, train_y = load_mnist(kind='train')
    test_x, test_y = load_mnist(kind='t10k')

    # Split and get codec list
    codec = codec.split(',')

    # Set model parameters
    model_parameter = ModelDNN(train_x=train_x, train_y=train_y,
                               test_x=test_x, test_y=test_y,
                               codec=codec,
                               psgd_type=psgd,
                               server_codec=server_codec,
                               learn_rate=lr,
                               epoches=epo,
                               optimizer_type=op,
                               block_assignment=assignment)

    assignment = model_parameter.get_assignment()(node_count, redundancy)
    GlobalSettings.set_default(node_count, redundancy, batch_size * assignment.block_count, assignment)

    core = Coordinator(model_parameter, logger)

    with open(arg.workers, 'r') as f:
        workers = json.load(f)

    core.set_workers(workers, node_count)

    try:
        if not arg.do_retrieve_only:
            core.resources_dispatch()
        else:
            core.require_client_log()
    except ConnectionAbortedError:
        print('All Done.')

