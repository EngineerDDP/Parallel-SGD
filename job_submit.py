from profiles.settings import GlobalSettings
from server_util.init_model import ModelDNN, ModelCNN, ModelLinear
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

    # Flags
    parse.add_argument("--retrieve", action="store_true", dest="do_retrieve_only", default=False, help="Set this flag to retrieve trace files from given workers.")
    parse.add_argument("--non-iid", action="store_true", default=False, dest="make_iid_dataset", help="Set this flag to make the dataset non-iid compatible.")
    parse.add_argument("--image-classification", action="store_true", default=True, dest="is_img_cls", help="Set this flag to indicate that target task is image classification.")

    # Shorter Sign
    parse.add_argument("-n", "--node-count", dest="n", type=int, default=1, help="Worker node count")
    parse.add_argument("-b", "--batch-size", dest="b", type=int, default=64, help="Batch size")
    parse.add_argument("-r", "--redundancy", dest="r", type=int, default=1, help="Redundancy")
    parse.add_argument("-C", "--codec", dest="codec", type=str, default='plain', help="Initial communication codec and protocol {ccdc, plain, ps}")
    parse.add_argument("-O", "--optimizer", dest="op", type=str, default='psgd', help="Set optimizer used for model training.")
    parse.add_argument("-E", "--epochs", dest="epochs", type=int, default=2, help="Train epochs")
    parse.add_argument("-D", "--dataset", dest="dataset", type=str, default='mnist', help="Dataset in use.")

    # Not commonly used
    parse.add_argument("--psgd", type=str, default='ssgd', help="Parallel stochastic gradient descent synchronization type {asgd, ssgd}")
    parse.add_argument("--learn-rate", dest="lr", type=float, default=0.05, help="Learining rate")
    parse.add_argument("--block-assignment", dest="assignment", type=str, default='iid', help="Block assignment strategy")
    parse.add_argument("--server-codec", dest="server_codec", type=str, default='grad', help="Server codec for parameter averaging")
    parse.add_argument("--workers", type=str, default='worker.json', help='Worker list file, json type')

    arg = parse.parse_args()

    logger = Logger(title_info='User Submit', log_to_file=True)

    logger.log_message('Initializing with parameters: ')
    logger.log_message('\t --node_count <node count {}>'.format(arg.n))
    logger.log_message('\t --batch_size <batch size {}>'.format(arg.b))
    logger.log_message('\t --redundancy <r {}>'.format(arg.r))
    logger.log_message('\t --codec <communication codec and protocol {}>'.format(arg.codec))
    logger.log_message('\t --optimizer <optimizer for model training {}>'.format(arg.op))
    logger.log_message('\t --epochs <training epochs {}>'.format(arg.epochs))
    logger.log_message('\t --dataset <using {}>'.format(arg.dataset))
    logger.log_message('\t --non-iid <{}>'.format(arg.make_iid_dataset))

    logger.log_message('\t --psgd <parallel stochastic gradient descent synchronization type {}>'.format(arg.psgd))
    logger.log_message('\t --learn_rate <learn rate for GD algorithm {}>'.format(arg.lr))
    logger.log_message('\t --block_assignment <block assignment strategy {}>'.format(arg.assignment))
    logger.log_message('\t --server_codec <parameter server codec {}>'.format(arg.server_codec))

    # Split and get codec list
    codec = arg.codec.split(',')
    # get assignment class for batch_size calculation
    from server_util.init_model import get_assignment
    ass = get_assignment(arg.assignment)
    assignment = ass(arg.n, arg.r)
    # set up full batch_size
    batch_size = arg.b * assignment.block_count
    GlobalSettings.set_default(arg.n, arg.r, batch_size, assignment)
    # get dataset
    if arg.dataset == 'mnist':
        from dataset.mnist_input import load
    elif arg.dataset == 'cifar':
        from dataset.cifar import load
    elif arg.dataset == 'simlin':
        from dataset.simdata import load
        arg.is_img_cls = False
    else:
        logger.log_error("Input dataset type cannot find any matches.")
        exit(1)

    # load dataset
    train_x, train_y, test_x, test_y = load()

    # make iid
    if arg.make_iid_dataset:
        from utils.partition_helper import make_non_iid_distribution
        train_x, train_y = make_non_iid_distribution(train_x, train_y, batch_size)

    # make format
    if arg.is_img_cls:
        from dataset.utils import make_image_scale, make_onehot
        train_x = make_image_scale(train_x)
        test_x = make_image_scale(test_x)
        train_y = make_onehot(train_y)
        test_y = make_onehot(test_y)

    # Set model parameters
    model_parameter = ModelDNN(train_x=train_x, train_y=train_y,
                               test_x=test_x, test_y=test_y,
                               codec=codec,
                               psgd_type=arg.psgd,
                               server_codec=arg.server_codec,
                               learn_rate=arg.lr,
                               epoches=arg.epochs,
                               optimizer_type=arg.op)



    core = Coordinator(model_parameter, logger)

    with open(arg.workers, 'r') as f:
        workers = json.load(f)

    core.set_workers(workers, arg.n)

    try:
        if not arg.do_retrieve_only:
            core.resources_dispatch()
        else:
            core.require_client_log()
    except ConnectionAbortedError:
        print('All Done.')

