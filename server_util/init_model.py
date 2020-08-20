from abc import ABCMeta, abstractmethod

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
    'qpa': FPWParaServer
}

def get_para_server(x: str):
    if __para_server_map.get(x) is not None:
        return __para_server_map[x]
    else:
        return codec_auto_search(x)

# ---------------------------------------------------------------------------------------------------------------

from nn.layers import FCLayer_v2
from nn.layers import MaxPool, Conv2dLayer, Reshape

from nn.metrics import CategoricalAccuracy, MeanSquareError


class IServerModel(metaclass=ABCMeta):

    def __init__(self,
                 train_x, train_y, test_x, test_y,
                 psgd_type,
                 optimizer_type,
                 server_codec,
                 epoches,
                 server_type='asgd',
                 target_acc=None,
                 learn_rate=0.05):
        self.__optimizer = get_optimizer(optimizer_type)
        self.__server_Codec = get_para_server(server_codec)
        self.__server_Type = get_psgd(server_type)
        self.__syncType = get_psgd(psgd_type)
        self.__learning_Rate = learn_rate
        self.__epoches = epoches
        self.__target_Accuracy = target_acc
        self.__training_Data = (train_x, train_y)
        self.__test_Data = (test_x, test_y)

    @abstractmethod
    def weights_types(self):
        pass

    @abstractmethod
    def getWeightsInit(self):
        pass

    @abstractmethod
    def loss_type(self):
        pass

    @abstractmethod
    def codec_ctrl(self):
        pass

    @abstractmethod
    def metric(self):
        pass

    def target_acc(self):
        return self.__target_Accuracy

    def psgd_type(self):
        return self.__syncType

    def psgd_server_codec(self):
        return self.__server_Codec

    def psgd_server_type(self):
        return self.__server_Type

    def optimizer_type(self):
        return self.__optimizer

    def epoches(self):
        return self.__epoches

    def learn_rate(self):
        return self.__learning_Rate

    def train_data(self):
        return self.__training_Data

    def eval_data(self):
        return self.__test_Data


class ModelLinear(IServerModel):

    def __init__(self,
                 train_x, train_y, test_x, test_y,
                 psgd_type,
                 optimizer_type,
                 server_codec,
                 epoches,
                 server_type='asgd',
                 target_acc=None,
                 learn_rate=0.05,
                 codec=None):

        super().__init__(train_x, train_y, test_x, test_y,
                         psgd_type, optimizer_type, server_codec, epoches,
                         server_type, target_acc, learn_rate)

        self.__nn = None
        layer_units = [1024]

        # init codec
        if codec is None:
            codec = ['ndc']
        elif len(codec) != len(layer_units):
            codec = codec[:1] * len(layer_units)

        # init layer
        self.__nn = [FCLayer_v2(layer_units[-1], act=get_activation("linear"))]

        # activated layer
        input_sample = self.train_data()[0][0].reshape([1, -1])

        for nn in self.__nn:
            input_sample = nn.F(input_sample)

        self.__codec = [get_codec(c_str) for c_str in codec]

    def weights_types(self):
        return ['w', 'b']

    def getWeightsInit(self):
        return self.__nn

    def loss_type(self):
        return get_loss("mse")

    def codec_ctrl(self):
        return self.__codec

    def metric(self):
        return [MeanSquareError()]


class ModelDNN(IServerModel):

    def __init__(self,
                 train_x, train_y, test_x, test_y,
                 psgd_type,
                 optimizer_type,
                 server_codec,
                 epoches,
                 server_type='asgd',
                 target_acc=None,
                 learn_rate=0.05,
                 codec=None):

        super().__init__(train_x, train_y, test_x, test_y,
                         psgd_type, optimizer_type, server_codec, epoches,
                         server_type, target_acc, learn_rate)

        self.__nn = None
        layer_units = [784, 784, 392, 196, 128, 10]

        # init codec
        if codec is None:
            codec = ['plain', 'plain', 'plain', 'plain', 'plain', 'plain']
        elif len(codec) != len(layer_units):
            codec = codec[:1] * len(layer_units)

        # init layer
        self.__nn = [FCLayer_v2(i, act=get_activation("tanh")) for i in layer_units[:-1]]

        self.__nn.append(FCLayer_v2(layer_units[-1], act=get_activation("softmax")))

        # activated layer
        input_sample = self.train_data()[0][0].reshape([1, -1])

        for nn in self.__nn:
            input_sample = nn.F(input_sample)

        self.__codec = [get_codec(c_str) for c_str in codec]

    # ---------- attributes ----------

    def weights_types(self):
        return ['w', 'b']

    def getWeightsInit(self):
        return self.__nn

    def loss_type(self):
        return get_loss("xentropy_softmax")

    def codec_ctrl(self):
        return self.__codec

    def metric(self):
        return [CategoricalAccuracy()]

    # ---------- attributes ----------


class ModelCNN(IServerModel):

    def __init__(self,
                 train_x, train_y, test_x, test_y,
                 psgd_type,
                 optimizer_type,
                 server_codec,
                 epoches,
                 server_type='asgd',
                 target_acc=None,
                 learn_rate=0.05,
                 codec=None):

        super().__init__(train_x, train_y, test_x, test_y,
                         psgd_type, optimizer_type, server_codec, epoches,
                         server_type, target_acc, learn_rate)

        self.__nn = []
        self.__nn.append(Conv2dLayer([5,5], 64, 'SAME', [1,1]))
        self.__nn.append(Conv2dLayer([5,5], 64, 'SAME', [1,1]))
        self.__nn.append(MaxPool([2,2]))
        self.__nn.append(Conv2dLayer([3,3], 64, 'SAME', [1,1]))
        self.__nn.append(Conv2dLayer([3,3], 64, 'SAME', [1,1]))
        self.__nn.append(MaxPool([2,2]))
        self.__nn.append(Reshape([12288]))
        self.__nn.append(FCLayer_v2(1024, act=get_activation('tanh')))
        self.__nn.append(FCLayer_v2(784, act=get_activation('tanh')))
        self.__nn.append(FCLayer_v2(10, act=get_activation('softmax')))

        if codec is None:
            codec = ['plain' for _ in self.__nn]
        elif len(codec) != len(self.__nn):
            codec = codec[:1] * len(self.__nn)

        self.__codec = [get_codec(c_str) for c_str in codec]

    def weights_types(self):
        return ['k', 'b']

    def getWeightsInit(self):
        return self.__nn

    def loss_type(self):
        return get_loss("xentropy_softmax")

    def codec_ctrl(self):
        return self.__codec

    def metric(self):
        return [CategoricalAccuracy()]