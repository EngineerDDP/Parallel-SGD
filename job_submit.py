from profiles.settings import GlobalSettings
from server_util.init_model import ModelDNN, ModelCNN
from dataset.mnist_input import load_mnist
from coordinator import Coordinator

import argparse
import json


if __name__ == '__main__':

    """
        Usage:
        python job_submit.py --node_count 4 --batch_size 128 --redundancy 2 --codec ccdc \
         --psgd ssgd --learn_rate 0.05 --epochs 10 --block_assignment iid \
         --server_codec graddiff --workers worker.json
    """
    parse = argparse.ArgumentParser()
    parse.add_argument("--node_count", type=int, default=3, help="initial node count")
    parse.add_argument("--batch_size", type=int, default=192, help="initial batch size")
    parse.add_argument("--redundancy", type=int, default=1, help="initial redundancy")
    parse.add_argument("--codec", type=str, default='sgq', help="initial communication codec and protocol {ccdc, plain, psclient}")
    parse.add_argument("--psgd", type=str, default='ssgd', help="parallel stochastic gradient descent synchronization type {asgd, ssgd}")
    parse.add_argument("--learn_rate", type=float, default=0.05, help="initial learining rate")
    parse.add_argument("--epochs", type=int, default=100, help="initial train epochs")
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


    print('\t --node_count <node count {}>'.format(node_count))
    print('\t --batch_size <batch size {}>'.format(batch_size))
    print('\t --redundancy <r {}>'.format(redundancy))
    print('\t --codec <communication codec and protocol {}>'.format(codec))
    print('\t --psgd <parallel stochastic gradient descent synchronization type {}>'.format(psgd))
    print('\t --learn_rate <learn rate for GD algorithm {}>'.format(lr))
    print('\t --epochs <training epochs {}>'.format(epo))
    print('\t --block_assignment <block assignment strategy {}>'.format(assignment))
    print('\t --server_codec <parameter server codec {}>'.format(server_codec))

    train_x, train_y = load_mnist(kind='train')
    test_x, test_y = load_mnist(kind='t10k')

    # Set model parameters
    model_parameter = ModelDNN(train_x=train_x, train_y=train_y,
                               test_x=test_x, test_y=test_y,
                               codec=codec,
                               psgd_type=psgd,
                               learn_rate=lr,
                               epoches=epo,
                               block_assignment=assignment)

    GlobalSettings.set_default(node_count, redundancy, batch_size, model_parameter.get_assignment())

    core = Coordinator(model_parameter)

    with open(arg.workers, 'r') as f:
        workers = json.load(f)

    core.set_workers(workers)

    core.require_client_log()

