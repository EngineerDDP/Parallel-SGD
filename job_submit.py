import argparse
import json

from dataset.transforms import TransformerList
from models.local.neural_models import ModelLinear
from models.trans import Req
# method to start up a network
from network import NodeAssignment, Request

from profiles import Settings
from roles import Coordinator, Reclaimer
from utils.constants import Parameter_Server
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
    parse.add_argument("--no-image-classification", action="store_false", default=True, dest="is_img_cls", help="Set this flag to indicate that target task is image classification.")

    # Shorter Sign
    parse.add_argument("-n", "--node-count", dest="n", type=int, default=1, help="Worker node count")
    parse.add_argument("-G", "--batch-size", dest="G", type=int, default=64, help="Batch size")
    parse.add_argument("-r", "--redundancy", dest="r", type=int, default=1, help="Redundancy")
    parse.add_argument("-C", "--codec", dest="codec", type=str, default='plain', help="Initial communication codec and protocol {ccdc, plain, ps}")
    parse.add_argument("-O", "--optimizer", dest="op", type=str, default='psgd', help="Set optimizer used for model training.")
    parse.add_argument("-E", "--epochs", dest="epochs", type=int, default=2, help="Train epochs")
    parse.add_argument("-D", "--dataset", dest="dataset", type=str, default='mnist', help="Dataset in use.")
    parse.add_argument("-I", "--input_ref", dest='input_shape', type=int, default=784, help="Input dimension.")

    # Not commonly used
    parse.add_argument("--psgd", type=str, default='ssgd', help="Parallel stochastic gradient descent synchronization type {asgd, ssgd}")
    parse.add_argument("--learn-rate", dest="lr", type=float, default=0.05, help="Learining rate")
    parse.add_argument("--block-assignment", dest="assignment", type=str, default='iid', help="Block assignment strategy")
    parse.add_argument("--server-codec", dest="server_codec", type=str, default='null', help="Server codec for parameter averaging")
    parse.add_argument("--workers", type=str, default='worker.json', help='Worker list file, json type')

    parse.add_argument("--network-bandwidth", dest='bandwidth', type=int, default=180, help='Network bandwidth in Mbps.')

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

    # get assignment class for batch_size calculation
    from models.local.neural_models import get_assignment
    ass = get_assignment(arg.assignment)
    assignment = ass(arg.n, arg.r)
    # set up full batch_size
    batch_size = arg.b * assignment.block_count
    setting = Settings(arg.n, arg.r, batch_size, assignment)
    dataset = None

    # get dataset
    if arg.dataset == 'cifar':
        from dataset.cifar import CIFAR
        dataset = CIFAR()
    elif arg.dataset == 'mnist':
        from dataset.mnist import MNIST
        dataset = MNIST()
    elif arg.dataset == 'simlin':
        from dataset.simdata import SimLin
        dataset = SimLin()
    else:
        logger.log_error("Input dataset type cannot find any matches.")
        exit(1)

    # get transform
    transform = TransformerList()
    if arg.make_iid_dataset:
        from dataset.transforms.non_iid_transform import Make_Non_IID
        transform.add(Make_Non_IID(batch_size))

    # make format
    if arg.is_img_cls:
        from dataset.transforms.image import ImageCls
        transform.add(ImageCls())

    # Split and get codec list
    codec = arg.codec.split(',')
    # Set model parameters
    model_parameter = ModelLinear(codec=codec,
                               psgd_type=arg.psgd,
                               server_codec=arg.server_codec,
                               learn_rate=arg.lr,
                               epoches=arg.epochs,
                               optimizer_type=arg.op,
                               input_shape=arg.input_shape)

    with open(arg.workers, 'r') as f:
        workers = json.load(f)

    pkg = NodeAssignment()
    i = 0

    if model_parameter.psgd_server_codec is not None:
        pkg.add(Parameter_Server, workers["PS"])
        logger.log_message("Add parameter server: address: ({})".format(workers["PS"]))

    for addr in workers["Worker"]:
        pkg.add(i, addr)
        logger.log_message("Add worker: id: ({}), address: ({})".format(i, addr))
        i += 1
        if i >= arg.n:
            break

    req = Request()
    with req.request(pkg) as com:
        if arg.do_retrieve_only:
            core = Reclaimer(com, logger)
            core.require_client_log()
        else:
            core = Coordinator(com, estimate_bandwidth=arg.bandwidth * 1024 * 1024, logger=logger)
            from executor.psgd.worker import PSGDPSExecutor, PSGDWorkerExecutor
            if model_parameter.psgd_server_codec is not None:
                core.submit_single(PSGDPSExecutor, worker_id=Parameter_Server, package_size=dataset.estimate_size())
            core.submit_group(PSGDWorkerExecutor, worker_offset=0, worker_cnt=arg.n, package_size=dataset.estimate_size())

            from models.trans.net_package import data_package, data_content, global_setting_package, essentials
            core.resources_dispatch({
                Req.GlobalSettings: global_setting_package(setting),
                Req.Weights_And_Layers: essentials(model_parameter),
                Req.Samples: data_content(dataset, transform),
                Req.Dataset: data_package(dataset, transform)
            })

            try:
                core.join()
            except ConnectionAbortedError:
                print('Worker exited without reported.')
