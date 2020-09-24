import numpy as np

from codec.essential import BlockWeight
from codec.interfaces import Codec, netEncapsulation
from utils.constants import Parameter_Server


def q1(arr: np.ndarray):
    std = np.std(arr)
    weights = arr.copy()
    weights[np.abs(weights) < std] = 0
    weights = np.sign(weights)
    return weights, std


def q2(arr: np.ndarray):
    std = np.std(arr)
    weights = (arr - np.mean(arr)) / std
    # be aware, this will change the value of referenced object.
    weights[weights > std] = 1
    weights[weights < -std] = -1
    weights[np.abs(weights) != 1] = 0
    return weights, std


class QuantizedPack:

    def __init__(self, node_id, content, std):
        self.node_id = node_id
        self.content = content
        self.std = std

    def pack(self):
        return [self.node_id, self.content, self.std]

    @staticmethod
    def unpack(content):
        return QuantizedPack(content[0], content[1], content[2])


class Quantization1BitPSCodec(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.epsilon = 0.01

    def dispose(self):
        pass

    def update_blocks(self, block_weight):
        weights, std = q1(block_weight.content)
        weights = weights.astype('int8')

        yield netEncapsulation(Parameter_Server, QuantizedPack(self.__node_id, weights, std).pack())

    def receive_blocks(self, content: list):
        content = QuantizedPack.unpack(content)
        self.set_result(content.content * content.std)


class Quantization2BitPSCodec(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)

        self.epsilon = 0.1

        self.Block_Weights_Std = 0
        self.Block_Weights_Mean = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight):
        weights, std = q2(block_weight.content)
        weights = weights.astype('int8')

        yield netEncapsulation(Parameter_Server, QuantizedPack(self.__node_id, weights, std).pack())

    def receive_blocks(self, content: list):
        content = QuantizedPack.unpack(content)
        self.set_result(content.content * content.std)


class FPWParaServer(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, content: list):
        content = QuantizedPack.unpack(content)
        self.__global_weights -= content.content.astype('double') * content.std
        yield netEncapsulation(content.node_id, QuantizedPack(Parameter_Server, self.__global_weights, 1).pack())


class LPWParaServer(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight:BlockWeight):
        pass

    def receive_blocks(self, content:list):
        pkg = QuantizedPack.unpack(content)
        self.__global_weights -= pkg.content.astype('double') * pkg.std
        return netEncapsulation(pkg.node_id, QuantizedPack(Parameter_Server, self.__global_weights.astype('float16'), 1).pack())


class Q2WParaServer(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight:BlockWeight):
        pass

    def receive_blocks(self, content:list):
        pkg = QuantizedPack.unpack(content)
        self.__global_weights -= pkg.content.astype('double') * pkg.std
        weights, std = q2(self.__global_weights)
        weights = weights.astype('int8')
        return netEncapsulation(pkg.node_id, QuantizedPack(Parameter_Server, weights, std).pack())


class Q1WParaServer(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, content: list):
        pkg = QuantizedPack.unpack(content)
        self.__global_weights -= pkg.content.astype('double') * pkg.std
        weights, std = q1(self.__global_weights)
        weights = weights.astype('int8')
        return netEncapsulation(pkg.node_id, QuantizedPack(Parameter_Server, weights, std).pack())

