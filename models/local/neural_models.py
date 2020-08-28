import numpy as np
from abc import ABCMeta, abstractmethod

from models.local.__init__ import IServerModel
from models.local.utils import *

from nn.layers import FCLayer_v2
from nn.layers import MaxPool, Conv2dLayer, Reshape

from nn.metrics import CategoricalAccuracy, MeanSquareError


class AbsServerModel(IServerModel, metaclass=ABCMeta):

    def __init__(self,
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

    @property
    @abstractmethod
    def weights_types(self):
        pass

    @abstractmethod
    def get_nn(self):
        pass

    @abstractmethod
    def loss_type(self):
        pass

    @property
    @abstractmethod
    def codec_ctrl(self):
        pass

    @property
    @abstractmethod
    def metric(self):
        pass

    @property
    def target_acc(self):
        return self.__target_Accuracy

    @property
    def psgd_type(self):
        return self.__syncType

    @property
    def psgd_server_codec(self):
        return self.__server_Codec

    @property
    def psgd_server_type(self):
        return self.__server_Type

    @property
    def optimizer_type(self):
        return self.__optimizer

    @property
    def epoches(self):
        return self.__epoches

    @property
    def learn_rate(self):
        return self.__learning_Rate


class ModelLinear(AbsServerModel):

    def __init__(self,
                 psgd_type,
                 optimizer_type,
                 server_codec,
                 epoches,
                 server_type='asgd',
                 target_acc=None,
                 learn_rate=0.05,
                 codec=None,
                 input_shape=784):

        super().__init__(psgd_type, optimizer_type, server_codec, epoches,
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
        input_sample = np.random.normal(size=[1, input_shape])

        for nn in self.__nn:
            input_sample = nn.F(input_sample)

        self.__codec = [get_codec(c_str) for c_str in codec]

    @property
    def weights_types(self):
        return ['w', 'b']

    def get_nn(self):
        return self.__nn

    @property
    def loss_type(self):
        return get_loss("mse")

    @property
    def codec_ctrl(self):
        return self.__codec

    @property
    def metric(self):
        return [MeanSquareError()]


class ModelDNN(AbsServerModel):

    def __init__(self,
                 psgd_type,
                 optimizer_type,
                 server_codec,
                 epoches,
                 server_type='asgd',
                 target_acc=None,
                 learn_rate=0.05,
                 codec=None,
                 input_shape=784):

        super().__init__(psgd_type, optimizer_type, server_codec, epoches,
                         server_type, target_acc, learn_rate)

        self.__nn = None
        layer_units = [784, 784, 784, 196, 128, 10]

        # init codec
        if codec is None:
            codec = ['plain'] * len(layer_units)
        elif len(codec) != len(layer_units):
            codec = codec[:1] * len(layer_units)
        codec.reverse()

        # init layer
        self.__nn = [FCLayer_v2(i, act=get_activation("tanh")) for i in layer_units[:-1]]

        self.__nn.append(FCLayer_v2(layer_units[-1], act=get_activation("softmax")))

        # activated layer
        input_sample = np.random.normal(size=[1, input_shape])

        for nn in self.__nn:
            input_sample = nn.F(input_sample)

        self.__codec = [get_codec(c_str) for c_str in codec]

    # ---------- attributes ----------

    @property
    def weights_types(self):
        return ['w', 'b']

    def get_nn(self):
        return self.__nn

    @property
    def loss_type(self):
        return get_loss("xentropy_softmax")

    @property
    def codec_ctrl(self):
        return self.__codec

    @property
    def metric(self):
        return [CategoricalAccuracy()]

    # ---------- attributes ----------


class ModelCNN(AbsServerModel):

    def __init__(self,
                 psgd_type,
                 optimizer_type,
                 server_codec,
                 epoches,
                 server_type='asgd',
                 target_acc=None,
                 learn_rate=0.05,
                 codec=None,
                 input_shape=None):

        super().__init__(psgd_type, optimizer_type, server_codec, epoches,
                         server_type, target_acc, learn_rate)

        if input_shape is None:
            input_shape = [32, 32, 3]

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

        input_sample = np.random.normal(size=[1] + input_shape)

        for nn in self.__nn:
            input_sample = nn.F(input_sample)

        self.__codec = [get_codec(c_str) for c_str in codec]

    @property
    def weights_types(self):
        return ['k', 'b']

    def get_nn(self):
        return self.__nn

    @property
    def loss_type(self):
        return get_loss("xentropy_softmax")

    @property
    def codec_ctrl(self):
        return self.__codec

    @property
    def metric(self):
        return [CategoricalAccuracy()]