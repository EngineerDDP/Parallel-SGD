import numpy as np

from codec.essential import Block_Weight
from utils.log import Logger
from utils.constants import Parameter_Server
from codec.interfaces import ICommunication_Ctrl, netEncapsulation


class QuantizedPack:

    def __init__(self, node_id, content, std):
        self.node_id = node_id
        self.content = content
        self.std = std


class Quantization1BitPSCodec(ICommunication_Ctrl):

    def __init__(self, node_id):
        super().__init__()
        self.__node_id = node_id
        self.epsilon = 0.01

    def dispose(self):
        pass

    def update_blocks(self, block_weight):
        weights = block_weight.Content
        std = np.std(weights)
        weights[np.abs(weights) < std] = 0
        weights = np.sign(weights)

        weights = weights.astype('int8')

        yield netEncapsulation(Parameter_Server, QuantizedPack(self.__node_id, weights, std))

    def receive_blocks(self, content: QuantizedPack):
        self.set_result(content.content)


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
        weights = block_weight.Content
        std = np.std(weights)
        weights = (weights - np.mean(weights)) / std

        # be aware, this will change the value of referenced object.
        weights[weights > std] = 1
        weights[weights < -std] = -1
        weights[np.abs(weights) != 1] = 0
        weights = weights.astype('int8')

        yield netEncapsulation(self.__node_id, QuantizedPack(self.__node_id, weights, std))

    def receive_blocks(self, content: QuantizedPack):
        self.set_result(content.content)


class FPWParaServer(ICommunication_Ctrl):

    def __init__(self, node_id):
        super().__init__()
        self.__global_weights = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: Block_Weight):
        pass

    def receive_blocks(self, content: QuantizedPack):
        self.__global_weights -= content.content.astype('double') * content.std
        yield netEncapsulation(content.node_id, QuantizedPack(Parameter_Server, self.__global_weights, 1))


