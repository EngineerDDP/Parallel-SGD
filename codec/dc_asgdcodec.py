import numpy as np

from profiles.settings import GlobalSettings
from codec.naive_ps import GradDiffParaServerCodec, PAServerCompack
from utils.log import Logger


class DCASGDServerCodec(GradDiffParaServerCodec):
    """
        Implemented according to paper:
        Shuxin Zheng, Qi Meng, Taifeng Wang, et al. Asynchronous Stochastic Gradient Descent with Delay Compensation.
        International Conference on Machine Learning (ICML), Sydney, Australia. 2017.
    """

    def __init__(self, node_id, logger=Logger('None')):

        GradDiffParaServerCodec.__init__(self, node_id, logger)

        # init weights
        self.Weights_init = 0

        # other parameters
        self.Learn_Rate = 1
        self.Bak_Weights_Node = {}
        self.Lambda_T = 2
        self.Mean_Square = 0
        self.Mean_Square_Epsilon = 1e-7
        self.Mean_Square_M = 0.95

        # init w_bak
        for key in GlobalSettings.get_default().nodes:
            self.Bak_Weights_Node[key] = self.Weights_init

    def receive_blocks(self, content:dict):
        """
            Adaptive DC-ASGD algorithm.
        """
        compack = PAServerCompack.from_dictionary(content)
        content = compack.Content
        content_square = np.multiply(content, content)
        # Update weights with delay-compensation
        delay = np.multiply(content_square, self.Weights_init - self.Bak_Weights_Node[self.Node_ID])
        self.Weights_init = self.Weights_init - self.Learn_Rate * (content + self.Lambda_T * delay)
        self.Mean_Square = self.Mean_Square_M * self.Mean_Square + (1 - self.Mean_Square_M) * content_square
        self.Lambda_T = self.Lambda_T / np.sqrt(self.Mean_Square + self.Mean_Square_Epsilon)
        # Send back updated weights
        self.Bak_Weights_Node[self.Node_ID] = self.Weights_init
        return self.Weights_init
