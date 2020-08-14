from abc import ABCMeta, abstractmethod

# register
# ---------------------------------------------------------------------------------------------------------------
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
    return __activation_map[x]

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
from codec.sgq import SGQClient

__codec_map = {
    'ccdc': CodedCommunicationCtrl,
    'plain': PlainCommunicationCtrl,
    'ps': PAClientCodec,
    'ndc': NaiveDuplicationCodec,
    'unicast': UnicastCommunicationCtrl,
    'q1': Quantization1BitPSCodec,
    'q2': Quantization2BitPSCodec,
    'sgq': SGQClient
}

def get_codec(x: str):
    return __codec_map[x]

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
from codec.sgq import SGQServer
from codec.quantization import FPWParaServer

__para_server_map = {
    'simple': ParaServerCodec,
    'grad': GradDiffParaServerCodec,
    'dc': DCASGDServerCodec,
    'sgq': SGQServer,
    'qpa': FPWParaServer
}

def get_para_server(x: str):
    return __para_server_map[x]

# ---------------------------------------------------------------------------------------------------------------

from nn.layers import FCLayer_v2
from nn.layers import MaxPool, Conv2dLayer, Reshape


class IServerModel(metaclass=ABCMeta):

    @abstractmethod
    def weights_types(self):
        pass

    @abstractmethod
    def getWeightsInit(self):
        pass

    @abstractmethod
    def initWeights(self):
        pass

    @abstractmethod
    def codec_ctrl(self):
        pass

    @abstractmethod
    def target_acc(self):
        pass

    @abstractmethod
    def psgd_type(self):
        pass

    @abstractmethod
    def psgd_server_codec(self):
        pass

    @abstractmethod
    def psgd_server_type(self):
        pass

    @abstractmethod
    def optimizer_type(self):
        pass

    @abstractmethod
    def loss_type(self):
        pass

    @abstractmethod
    def epoches(self):
        pass

    @abstractmethod
    def learn_rate(self):
        pass

    @abstractmethod
    def train_data(self):
        pass

    @abstractmethod
    def eval_data(self):
        pass

    @abstractmethod
    def get_assignment(self):
        pass


class ModelDNN(IServerModel):

    def __init__(self,
                 train_x, train_y, test_x, test_y,
                 layer_units=None,
                 activation='tanh',
                 output='softmax',
                 loss='xentropy_softmax',
                 learn_rate=0.05,
                 codec=None,
                 psgd_type='ssgd',
                 optimizer_type='psgd',
                 server_type='asgd',
                 server_codec='grad',
                 epoches=10,
                 target_acc=None,
                 block_assignment='iid'
                 ):
        if layer_units is None:
            layer_units = [784, 784, 392, 196, 128, 10]
        if codec is None:
            codec = ['plain', 'plain', 'plain', 'plain', 'plain', 'plain']
        elif len(codec) == 1:
            codec = codec * len(layer_units)

        self.Layer_Units = layer_units
        self.Activation = get_activation(activation)
        self.Activation_out = get_activation(output)
        self.Loss = get_loss(loss)
        self.Codec = [get_codec(c_str) for c_str in codec]
        self.Optimizer = get_optimizer(optimizer_type)
        self.Server_Codec = get_para_server(server_codec)
        self.Server_Type = get_psgd(server_type)
        self.SyncType = get_psgd(psgd_type)
        self.Learning_Rate = learn_rate
        self.Epoches = epoches
        self.Target_Accuracy = target_acc
        self.Training_Data = (train_x, train_y)
        self.Test_Data = (test_x, test_y)
        self.Neural_Network = None
        self.Block_Assignment = get_assignment(block_assignment)
        self.initWeights()

    # ---------- attributes ----------

    def weights_types(self):
        return ['w', 'b']

    def getWeightsInit(self):
        return self.Neural_Network

    def codec_ctrl(self):
        return self.Codec

    def target_acc(self):
        return self.Target_Accuracy

    def psgd_type(self):
        return self.SyncType

    def psgd_server_codec(self):
        return self.Server_Codec

    def psgd_server_type(self):
        return self.Server_Type

    def optimizer_type(self):
        return self.Optimizer

    def loss_type(self):
        return self.Loss

    def epoches(self):
        return self.Epoches

    def learn_rate(self):
        return self.Learning_Rate

    def train_data(self):
        return self.Training_Data

    def eval_data(self):
        return self.Test_Data

    def get_assignment(self):
        return self.Block_Assignment

    # ---------- attributes ----------

    def initWeights(self):

        self.Neural_Network = []

        for units in self.Layer_Units[:-1]:
            self.Neural_Network.append(FCLayer_v2(units, act=self.Activation()))

        self.Neural_Network.append(FCLayer_v2(self.Layer_Units[-1], act=self.Activation_out()))

        # activated layer
        input_sample = self.Training_Data[0][0].reshape([1, -1])

        for nn in self.Neural_Network:
            input_sample = nn.F(input_sample)

        assert len(self.Neural_Network) == len(self.Codec), "{} codec required, but got {}.".format(len(self.Neural_Network), len(self.Codec))

        return


class ModelCNN(IServerModel):

    def weights_types(self):
        pass

    def psgd_server_codec(self):
        pass

    def psgd_server_type(self):
        pass

    def optimizer_type(self):
        pass

    def getWeightsInit(self):
        pass

    def initWeights(self):
        pass

    def codec_ctrl(self):
        pass

    def target_acc(self):
        pass

    def psgd_type(self):
        pass

    def loss_type(self):
        pass

    def epoches(self):
        pass

    def learn_rate(self):
        pass

    def train_data(self):
        pass

    def eval_data(self):
        pass