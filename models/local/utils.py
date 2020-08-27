# register
# ---------------------------------------------------------------------------------------------------------------
def codec_auto_search(abs_name: str):
    import importlib
    try:
        package_name, class_name = abs_name.split('.')
        mod = importlib.import_module('codec.' + package_name)
        cls = getattr(mod, class_name)
    except Exception:
        raise AssertionError('Cannot find codec \'{}\'.'.format(abs_name))
    return cls

from nn.activations import Sigmoid
from nn.activations import Tanh, Linear, ReLU, SoftmaxNoGrad

__activation_map = {
    'tanh': Tanh,
    'sigmoid': Sigmoid,
    'linear': Linear,
    'relu': ReLU,
    'softmax': SoftmaxNoGrad
}

def get_activation(x: str):
    return __activation_map[x]()

from nn.losses import CrossEntropyLoss, CrossEntropyLossWithSoftmax
from nn.losses import MseLoss

__loss_map = {
    'mse': MseLoss,
    'xentropy': CrossEntropyLoss,
    'xentropy_softmax': CrossEntropyLossWithSoftmax
}

def get_loss(x: str):
    return __loss_map[x]

from codec.ccdc import CodedCommunicationCtrl
from codec.naive_ps import PAClientCodec
from codec.plain import PlainCommunicationCtrl
from codec.ndc import NaiveDuplicationCodec
from codec.unicast import UnicastCommunicationCtrl
from codec.quantization import Quantization1BitPSCodec, Quantization2BitPSCodec

__codec_map = {
    'ccdc': CodedCommunicationCtrl,
    'plain': PlainCommunicationCtrl,
    'ps': PAClientCodec,
    'ndc': NaiveDuplicationCodec,
    'unicast': UnicastCommunicationCtrl,
    'q1': Quantization1BitPSCodec,
    'q2': Quantization2BitPSCodec
}

def get_codec(x: str):
    if __codec_map.get(x) is not None:
        return __codec_map[x]
    else:
        return codec_auto_search(x)

from nn.optimizer import ParallelSGDOptimizer, ParaAverageOptimizer, FastParallelSGDOptimizer

__optimizer_map = {
    'psgd': ParallelSGDOptimizer,
    'pa': ParaAverageOptimizer,
    'fast': FastParallelSGDOptimizer
}

def get_optimizer(x: str):
    return __optimizer_map[x]

from psgd.asgd import AsynchronizedSGD
from psgd.ssgd import SynchronizedSGD

__psgd_map = {
    'ssgd': SynchronizedSGD,
    'asgd': AsynchronizedSGD
}

def get_psgd(x: str):
    return __psgd_map[x]

from profiles.blockassignment.idependent import IIDBlockAssignment
from profiles.blockassignment.duplicate import DuplicateAssignment

__assignment_map = {
    'iid': IIDBlockAssignment,
    'dpa': DuplicateAssignment
}

def get_assignment(x: str):
    return __assignment_map[x]

from codec.naive_ps import ParaServerCodec, GradDiffParaServerCodec
from codec.dc_asgdcodec import DCASGDServerCodec
from codec.quantization import FPWParaServer

__para_server_map = {
    'simple': ParaServerCodec,
    'grad': GradDiffParaServerCodec,
    'dc': DCASGDServerCodec,
    'qpa': FPWParaServer,
    'null': None
}

def get_para_server(x: str):
    if __para_server_map.get(x, 'null') is not 'null':
        return __para_server_map[x]
    else:
        return codec_auto_search(x)

# ---------------------------------------------------------------------------------------------------------------
