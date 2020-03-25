import numpy as np

from log import Logger
from codec.pacodec import PAClientCodec


class Quantization1BitPSCodec(PAClientCodec):

    def __init__(self, node_id, logger=Logger('None')):
        super().__init__(node_id, logger)

        self.beta = 0.9
        self.gamma = 0.9
        self.epsilon = 0.01

        self.Block_Weights_Std = 0
        self.Block_Weights_Mean = 0

    def update_blocks(self, block_weight):
        weights = block_weight.Content.copy()
        self.Block_Weights_Std = np.std(weights)
        self.Block_Weights_Mean = np.mean(weights)

        weights = weights - self.Block_Weights_Mean

        # be aware, this will change the value of inherited object.
        weights[weights > 0] = 1
        weights[weights < 0] = 0
        weights = weights.astype('int8')

        # useless code, ndarray assigned with reference.
        block_weight.Content = weights

        return super().update_blocks(block_weight)

    def receive_blocks(self, json_dict):
        super().receive_blocks(json_dict)
        weights = self.get_result().astype('double')
        weights[weights == 1] = 1
        weights[weights == 0] = -1
        weights = weights * self.Block_Weights_Std + self.Block_Weights_Mean
        self.set_result(weights)


class Quantization2BitPSCodec(PAClientCodec):

    def __init__(self, node_id, logger=Logger('None')):
        super().__init__(node_id, logger)

        self.epsilon = 0.1

        self.Block_Weights_Std = 0
        self.Block_Weights_Mean = 0

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

        # useless code, ndarray assigned with reference.
        block_weight.Content = weights

        return super().update_blocks(block_weight)

    def receive_blocks(self, json_dict):
        super().receive_blocks(json_dict)
        weights = self.get_result().astype('double')
        weights = weights * self.Block_Weights_Std + self.Block_Weights_Mean
        self.set_result(weights)


