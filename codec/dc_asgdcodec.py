import numpy as np

from codec.interfaces import ICommunicationCtrl, IComPack
from codec.pacodec import PAClientCodec, PAServerCodec, PAServerCompack

from network.agreements import DefaultNodes

from server_util.init_model import ModelMNIST
from settings import GlobalSettings

from log import Logger

class DC_ASGDClientCodec(PAClientCodec):

    def __init__(self, node_id, logger=Logger('None')):

        PAClientCodec.__init__(self, node_id, logger)


class DC_ASGDServerCodec(PAServerCodec):
    """
        Implemented according to paper:
        Shuxin Zheng, Qi Meng, Taifeng Wang, et al. Asynchronous Stochastic Gradient Descent with Delay Compensation.
        International Conference on Machine Learning (ICML), Sydney, Australia. 2017.
    """

    def __init__(self, node_id, logger=Logger('None')):

        PAServerCodec.__init__(self, node_id, logger)

        # init weights
        self.Weights_init = 0

        # other parameters
        self.Learn_Rate = ModelMNIST.learn_rate()
        self.Bak_Weights_Node = {}
        self.Lambda_T = 2
        self.Mean_Square = 0
        self.Mean_Square_Epsilon = 1e-7
        self.Mean_Square_M = 0.95

        # init w_bak
        for key in GlobalSettings.get_default().Nodes:
            self.Bak_Weights_Node[key] = self.Weights_init

    def receive_blocks(self, json_dict):
        """
            Adaptive DC-ASGD algorithm.
        """
        compack = PAServerCompack.decompose_compack(json_dict)
        content = compack.Content
        # Update weights with delay-compensation
        delay = np.multiply(np.multiply(content, content), self.Weights_init - self.Bak_Weights_Node[self.Node_No])
        self.Weights_init = self.Weights_init - self.Learn_Rate * (content + self.Lambda_T * delay)
        self.Mean_Square = self.Mean_Square_M * self.Mean_Square + (1 - self.Mean_Square_M) * np.multiply(content, content)
        self.Lambda_T = self.Lambda_T / np.sqrt(self.Mean_Square + self.Mean_Square_Epsilon)
        # Send back updated weights
        self.Bak_Weights_Node[self.Node_No] = self.Weights_init
        return self.Weights_init