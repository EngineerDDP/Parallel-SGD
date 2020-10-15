import argparse
from typing import Type

import nn
from codec.interfaces import Codec
from dataset.interfaces import AbsDataset
from dataset.transforms import *
from executor.psgd import ParallelSGD, parse_worker
from nn import IOptimizer
from nn.gradient_descent.interface import IGradientDescent
from profiles.blockassignment.abstract import AbsBlockAssignment
from psgd.sync import IParallelSGD
from utils.log import Logger


def codec_auto_search(abs_name: str):
    import importlib
    try:
        package_name, class_name = abs_name.split('.')
        mod = importlib.import_module('codec.' + package_name)
        cls = getattr(mod, class_name)
    except Exception:
        raise AssertionError('Cannot find codec \'{}\'.'.format(abs_name))
    return cls


def codec_parser(name) -> [Type[Codec]]:
    if name == 'plain':
        from codec.plain import Plain
        return Plain
    elif name == 'ps_client':
        from codec.naive_ps import PSClient
        return PSClient
    elif name == 'grad_diff':
        from codec.naive_ps import GradDiffPS
        return GradDiffPS
    elif name == 'para_average':
        from codec.naive_ps import ParaAveragingServer
        return ParaAveragingServer
    elif name == 'dc_asgd':
        from codec.dc_asgdcodec import DCASGDServerCodec
        return DCASGDServerCodec
    elif name == 'binarySGD':
        from codec.quantization import Quantization1BitPSCodec
        return Quantization1BitPSCodec
    elif name == 'ternarySGD':
        from codec.quantization import Quantization2BitPSCodec
        return Quantization2BitPSCodec
    elif name == 'stochastic_ternary_SGD':
        from codec.tqn import TQNClient
        return TQNClient
    elif name == '1bit_weights_averaging':
        from codec.quantization import Q1WParaServer
        return Q1WParaServer
    elif name == '2bit_weights_averaging':
        from codec.quantization import Q2WParaServer
        return Q2WParaServer
    elif name == 'low_precision_weights_averaging':
        from codec.quantization import LPWParaServer
        return LPWParaServer
    elif name == 'high_precision_weights_averaging':
        from codec.quantization import FPWParaServer
        return FPWParaServer
    elif name == 'null':
        return None
    else:
        return codec_auto_search(name)


def sync_type_parser(name) -> Type[IParallelSGD]:
    if name == 'ssgd':
        from psgd.sync import SynchronizedSGD
        return SynchronizedSGD
    elif name == 'asgd':
        from psgd.sync import AsynchronizedSGD
        return AsynchronizedSGD


def optimizer_parser(name) -> Type[IOptimizer]:
    if name == 'gradient_averaging':
        from nn.optimizer import PSGDOptimizer
        return PSGDOptimizer
    elif name == 'double_buffering_gradient_averaging':
        from nn.optimizer import DoubleBufferingOptimizer
        return DoubleBufferingOptimizer
    elif name == 'parameter_averaging':
        from nn.optimizer import PAOptimizer
        return PAOptimizer


def gradient_descent_parser(name) -> Type[IGradientDescent]:
    if name == 'adam':
        from nn.gradient_descent import ADAMOptimizer
        return ADAMOptimizer
    elif name == 'sgd':
        from nn.gradient_descent import SGDOptimizer
        return SGDOptimizer
    elif name == 'adadelta':
        from nn.gradient_descent import AdaDeltaOptimizer
        return AdaDeltaOptimizer
    elif name == 'adagrad':
        from nn.gradient_descent import AdaGradOptimizer
        return AdaGradOptimizer
    elif name == 'rmsprop':
        from nn.gradient_descent import RMSPropOptimizer
        return RMSPropOptimizer


def block_assignment_parser(name) -> Type[AbsBlockAssignment]:
    if name == 'iid':
        from profiles.blockassignment import IIDBlockAssignment
        return IIDBlockAssignment
    elif name == 'dup':
        from profiles.blockassignment import DuplicateAssignment
        return DuplicateAssignment


def dataset_parser(name) -> [AbsDataset]:
    if name == 'cifar':
        from dataset.cifar import CIFAR
        return CIFAR()
    elif name == 'mnist':
        from dataset.mnist import MNIST
        return MNIST()
    else:
        return None


if __name__ == '__main__':

    """
        Usage:
        python job_submit.py --node_count 4 --batch_size 128 --redundancy 2 --codec ccdc \
         --psgd ssgd --learn_rate 0.05 --epochs 10 --block_assignment iid \
         --server_codec graddiff --workers worker.json
    """
    parse = argparse.ArgumentParser()

    # Flags
    parse.add_argument("--non-iid", action="store_true", default=False, dest="make_iid_dataset", help="Set this flag to make the dataset non-iid compatible.")

    # Shorter Sign
    parse.add_argument("-m", "--model", dest="model_file", required=True, type=str, help="Model file")
    parse.add_argument("-n", "--node-count", dest="n", type=int, default=1, help="Worker node count")
    parse.add_argument("-b", "--batch-size", dest="b", type=int, default=64, help="Batch size")
    parse.add_argument("-r", "--redundancy", dest="r", type=int, default=1, help="Redundancy")
    parse.add_argument("-C", "--codec", dest="codec", type=str, default='plain', help="Initial communication codec and protocol {ccdc, plain, ps}")
    parse.add_argument("-O", "--optimizer", dest="op", type=str, default='gradient_averaging', help="Set optimizer used for model training.")
    parse.add_argument("-E", "--epochs", dest="epochs", type=int, default=2, help="Train epochs")
    parse.add_argument("-D", "--dataset", dest="dataset", type=str, default='mnist', help="Dataset in use.")

    # Not commonly used
    parse.add_argument("--gradient-descent", type=str, default='sgd', dest='gd', help="Gradient optimizing method used in training")
    parse.add_argument("--psgd", type=str, default='ssgd', help="Parallel stochastic gradient descent synchronization type {asgd, ssgd}")
    parse.add_argument("--learn-rate", dest="lr", type=float, default=0.00005, help="Learining rate")
    parse.add_argument("--block-assignment", dest="assignment", type=str, default='iid', help="Block assignment strategy")
    parse.add_argument("--server-codec", dest="server_codec", type=str, default='null', help="Server codec for parameter averaging")
    parse.add_argument("--workers", type=str, default='worker.json', help='Worker list file, json type')

    parse.add_argument("--network-bandwidth", dest='bandwidth', type=int, default=100, help='Network bandwidth in Mbps.')

    arg = parse.parse_args()

    logger = Logger(title_info='P-SGD User Submit', log_to_file=True)

    logger.log_message('Initializing with parameters: ')
    logger.log_message('\t --model <file name {}>'.format(arg.model_file))
    logger.log_message('\t --node_count <node count {}>'.format(arg.n))
    logger.log_message('\t --batch_size <batch size {}>'.format(arg.b))
    logger.log_message('\t --redundancy <r {}>'.format(arg.r))
    logger.log_message('\t --codec <communication codec and protocol {}>'.format(arg.codec))
    logger.log_message('\t --optimizer <optimizer for model training {}>'.format(arg.op))
    logger.log_message('\t --epochs <training epochs {}>'.format(arg.epochs))
    logger.log_message('\t --dataset <using {}>'.format(arg.dataset))
    logger.log_message('\t --non-iid <{}>'.format(arg.make_iid_dataset))

    logger.log_message('\t --gradient-descent <Gradient method {}>'.format(arg.gd))
    logger.log_message('\t --psgd <parallel stochastic gradient descent synchronization type {}>'.format(arg.psgd))
    logger.log_message('\t --learn_rate <learn rate for GD algorithm {}>'.format(arg.lr))
    logger.log_message('\t --block_assignment <block assignment strategy {}>'.format(arg.assignment))
    logger.log_message('\t --server_codec <parameter server codec {}>'.format(arg.server_codec))

    # get dataset
    dataset = dataset_parser(arg.dataset)
    assert isinstance(dataset, AbsDataset), "Input dataset type cannot find any matches."

    # get transfer
    trans = Shuffle().add(ImageCls())

    # get model
    model = nn.model.Model.load(arg.model_file)
    assert model.can_fit, "Model is not prepared for training."

    # get nodes
    nodes = parse_worker(worker_cnt=arg.n, ps=(arg.server_codec != 'null'), filename=arg.workers)

    # get job
    job = ParallelSGD(model, dataset, trans)

    # run
    try:
        job.parallel(nodes,
                     redundancy=arg.r,
                     block_size=arg.b,
                     epoch=arg.epochs,
                     assignment_type=block_assignment_parser(arg.assignment),
                     sync_type=sync_type_parser(arg.psgd),
                     op_type=optimizer_parser(arg.op),
                     gd_type=gradient_descent_parser(arg.gd),
                     gd_params=(arg.lr, ),
                     codec=codec_parser(arg.codec),
                     ps_codec=codec_parser(arg.server_codec),
                     network_bandwidth=arg.bandwidth * 1048576,
                     mission_title="PSGDSubmit"
                     )
    except ConnectionAbortedError:
        print("Worker exited without reports.")
