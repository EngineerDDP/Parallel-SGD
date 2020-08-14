import numpy as np

from utils.log import Logger
from utils.constants import Parameter_Server
from codec.interfaces import ICommunication_Ctrl, netEncapsulation


class Quantization1BitPSCodec(ICommunication_Ctrl):

    def __init__(self, node_id):
        super().__init__()

        self.__node_id = node_id
        self.beta = 0.9
        self.gamma = 0.9
        self.epsilon = 0.01

        self.Block_Weights_Std = 0
        self.Block_Weights_Mean = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight):
        weights = block_weight.Content.copy()
        self.Block_Weights_Std = np.std(weights)
        self.Block_Weights_Mean = np.mean(weights)

        weights = weights - self.Block_Weights_Mean

        # be aware, this will change the value of inherited object.
        weights[weights > 0] = 1
        weights[weights < 0] = 0
        weights = weights.astype('bool')

        yield netEncapsulation(Parameter_Server, QuantizedPack(self.__node_id, weights))

    def receive_blocks(self, json_dict):
        super().receive_blocks(json_dict)
        weights = self.get_result().astype('double')
        weights[weights == 1] = 1
        weights[weights == 0] = -1
        weights = weights * self.Block_Weights_Std + self.Block_Weights_Mean
        self.set_result(weights)


class Quantization2BitPSCodec(ICommunication_Ctrl):

    def __init__(self, node_id):
        super().__init__()

        self.__node_id = node_id
        self.epsilon = 0.1

        self.Block_Weights_Std = 0
        self.Block_Weights_Mean = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight):
        weights = block_weight.Content.copy()
        self.Block_Weights_Std = np.std(weights)
        self.Block_Weights_Mean = np.mean(weights)

        weights = (weights - self.Block_Weights_Mean) / self.Block_Weights_Std

        # be aware, this will change the value of inherited object.
        weights[weights > self.epsilon] = 1
        weights[weights < -self.epsilon] = -1
        weights[np.abs(weights) != 1] = 0
        weights = weights.astype('int8')

        yield netEncapsulation(self.__node_id, weights)

    def receive_blocks(self, json_dict):
        super().receive_blocks(json_dict)
        weights = self.get_result().astype('double')
        weights = weights * self.Block_Weights_Std + self.Block_Weights_Mean
        self.set_result(weights)


class QuantizedPack:

    def __init__(self, id, content):
        self.Content = content
        self.Id = id

    def pack(self):
        return [self.Id, self.Content]

    @staticmethod
    def unpack(content:list):
        return QuantizedPack(content[0], content[1])

