from typing import Sequence

import numpy as np

from codec.essential import BlockWeight
from codec.interfaces import Codec, netEncapsulation
from utils.constants import Parameter_Server


def build_quantization_space(bits: int = 2) -> Sequence[int]:
    max_v = (1 << bits - 1) - 1
    min_v = -max_v
    return np.arange(min_v, max_v + 1, 1)


q_space = build_quantization_space(3)


def deterministic_quantization(arr: np.ndarray, space: Sequence[int]):
    arr[arr > space[-1]] = space[-1]
    arr[arr < space[0]] = space[0]
    return np.round(arr)


def stochastic_quantization(arr: np.ndarray, space: list):
    """
        Alistarh et al. “QSGD: Communication-Efficient SGD via Gradient Quantization
        and Encoding”. NIPS2017
    :param arr: array to be quantized.
    :param space: quantization space.
    :return: the base vector or quantized array.
    """
    arr_in = np.abs(arr)
    for x in np.nditer(arr_in, op_flags=["readwrite"]):
        lo = 0.0
        hi = 0.0
        for i in space:
            if i < x:
                lo = i
            else:
                hi = i
                break

        rnd = np.random.uniform(lo, hi)
        if rnd > x:
            x[...] = lo
        else:
            x[...] = hi

    return np.sign(arr) * arr_in


def quantize(arr, space, epsilon: float = 1e-9, iterations: int = 3):
    """
        TERNARIZATION TQN implementation based on chapter 3.1.1 in
        L. Hou and J. T. Kwok. Loss-aware weight quantization of deep networks.
        In International Conference on Learning Representations (ICLR), 2018.
    :param arr: array to be quantized.
    :param space: quantization space.
    :param epsilon: epsilon for the denominator of operation.
    :param iterations: iterations for calculation.
    :return: alpha, b.  alpha for the coefficient of quantized array, b for the base
        vector or quantized array.
    """
    a = 0.7
    b = deterministic_quantization(arr / a, space)
    for i in range(iterations):
        a = np.sum(np.multiply(b, arr)) / (np.sum(np.square(b)) + 1)
        b = deterministic_quantization(arr / (a + epsilon), space)
    return a, b


def binarize(arr: np.ndarray):
    """
        M. Courbariaux, Y. Bengio, and J. P. David. BinaryConnect:
        Training deep neural networks with binary weights during propagations.
        In Advances in Neural Information Processing Systems,
        pp.3105–3113, 2015.
    :param arr: arr to be binarized.
    :return: alpha, b.  alpha for the coefficient of quantized array, b for the base
        vector or quantized array.
    """
    std = np.std(arr)
    weights = np.sign(arr)
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


class BinaryNetCodec(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.epsilon = 0.01

    def dispose(self):
        pass

    def update_blocks(self, block_weight):
        weights, std = binarize(block_weight.content)
        # Use int8 for fast serialization.
        weights = weights.astype('int8')

        return netEncapsulation(Parameter_Server, QuantizedPack(self.node_id, weights, std).pack())

    def receive_blocks(self, content: list):
        content = QuantizedPack.unpack(content)
        self.set_result(content.content * content.std)


class QuantizedClient(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)

        self.epsilon = 0.1

        self.Block_Weights_Std = 0
        self.Block_Weights_Mean = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight):
        weights, std = quantize(block_weight.content, q_space)
        # Use int8 for fast serialization.
        weights = weights.astype('int8')

        return netEncapsulation(Parameter_Server, QuantizedPack(self.node_id, weights, std).pack())

    def receive_blocks(self, content: list):
        content = QuantizedPack.unpack(content)
        self.set_result(content.content * content.std)


class FullPrecisionParaServer(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, content: list):
        content = QuantizedPack.unpack(content)
        # Full precision uses float64 for transmission.
        self.__global_weights -= content.content.astype('double') * content.std
        return netEncapsulation(content.node_id, QuantizedPack(Parameter_Server, self.__global_weights, 1).pack())


class LowPrecisionParaServer(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, content: list):
        pkg = QuantizedPack.unpack(content)
        # Low precision uses float16 for transmission
        self.__global_weights -= (pkg.content.astype('double') * pkg.std).astype('float16')
        return netEncapsulation(pkg.node_id, QuantizedPack(Parameter_Server, self.__global_weights, 1).pack())


class QuantizedParaServer(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights: np.ndarray = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, content: list):
        pkg = QuantizedPack.unpack(content)
        # apply quantized gradients
        self.__global_weights -= pkg.content.astype('double') * pkg.std
        # quantize global weights
        weights, std = quantize(self.__global_weights, q_space)
        # Use int8 for fast serialization.
        weights = weights.astype('int8')
        return netEncapsulation(pkg.node_id, QuantizedPack(Parameter_Server, weights, std).pack())


class BinaryParaServer(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights: np.ndarray = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, content: list):
        pkg = QuantizedPack.unpack(content)
        # apply quantized gradients
        self.__global_weights -= pkg.content.astype('double') * pkg.std
        # quantize global weights
        weights, std = binarize(self.__global_weights)
        # Use int8 for fast serialization.
        weights = weights.astype('int8')
        return netEncapsulation(pkg.node_id, QuantizedPack(Parameter_Server, weights, std).pack())
