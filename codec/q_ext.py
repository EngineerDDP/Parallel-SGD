from abc import ABCMeta, abstractmethod
from typing import Sequence, Tuple

import numpy as np

from utils.quantize import quant
from codec.essential import BlockWeight
from codec.interfaces import Codec, netEncapsulation
from utils.constants import Parameter_Server


def build_quantization_space(states: int) -> Sequence[float]:
    """
        Build quantization space
    :param states: bits of input
    :return: list of float
    """
    k_max = (1 << (states - 1)) - 1
    def theta(k): return 1 / (np.tan(k * np.pi / 4))
    space_pos = [theta(k / k_max) for k in range(k_max, 0, -1)]
    space_neg = [-i for i in space_pos]
    space_neg.reverse()
    res = np.asarray(space_neg + [0] + space_pos).round(4)
    return res


class IPack(metaclass=ABCMeta):

    @property
    @abstractmethod
    def content(self):
        pass

    @property
    @abstractmethod
    def node_id(self):
        pass


class QuantizedPack(IPack):

    q = quant(build_quantization_space(3))

    @staticmethod
    def quantize(arr, epsilon: float = 1e-9, iterations: int = 3) -> Tuple[float, np.ndarray]:
        """
            Customized function
        :param arr: array to be quantized.
        :param epsilon: epsilon for the denominator of operation.
        :param iterations: iterations for calculation.
        :return: alpha, b.  alpha for the coefficient of quantized array, b for the base
            vector or quantized array.
        """
        a = 0.7
        binaries = QuantizedPack.q.stochastic(arr / a)
        for i in range(iterations):
            b = QuantizedPack.q.decode(binaries)
            a = np.sum(np.multiply(b, arr)) / (np.sum(np.square(b)) + 1)
            binaries = QuantizedPack.q.stochastic(arr / (a + epsilon))
        return a, binaries

    def __init__(self, node_id: int, array: np.ndarray):
        self.__node_id = node_id
        self.__shape = array.shape
        # Use int8 for fast serialization.
        self.__alpha, self.__b = QuantizedPack.quantize(array.reshape(-1))

    @property
    def content(self):
        return (self.__alpha * QuantizedPack.q.decode(self.__b)).reshape(self.__shape)

    @property
    def node_id(self):
        return self.__node_id


class QuantizedClient(Codec):

    q_space = build_quantization_space(6)

    def __init__(self, node_id):
        super().__init__(node_id)

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight) -> netEncapsulation[QuantizedPack]:
        package = QuantizedPack(self.node_id, block_weight.content)
        return netEncapsulation(Parameter_Server, package)

    def receive_blocks(self, package: IPack):
        self.set_result(package.content, lambda x, y: y)


class QuantizedParaServer(Codec):

    q_space = build_quantization_space(6)

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights: np.ndarray = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, package: IPack):
        self.__global_weights -= package.content
        reply = QuantizedPack(Parameter_Server, self.__global_weights)
        return netEncapsulation(package.node_id, reply)
