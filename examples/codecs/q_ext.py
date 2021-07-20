from abc import ABCMeta, abstractmethod
from typing import Sequence, Tuple

import numpy as np
from parallel_sgd.codec.utils.quantize import quant

from parallel_sgd.codec import GlobalSettings
from parallel_sgd.codec.essential import BlockWeight
from parallel_sgd.codec.interfaces import Codec, netEncapsulation
from constants import Parameter_Server

"""
    This codec requires specified parameters.
    Listed as below:
"""

Quantization_Resolution_Client = "QC"
Quantization_Resolution_Server = "QS"

"""
    Parameters listed above should be added to GlobalSettings.global_parameters as dict type.
    Fill the parameter "codec_extra_parameters" while calling rpc.parallel_sgd.submit.ParallelSGD.parallel_computing()
    with this codec.
"""


def build_quantization_space(states: int) -> Sequence[float]:
    """
        Build quantization space
    :param states: bits of input
    :return: list of float
    """
    states = max(int(states), 3)
    k_max = (1 << (states - 1)) - 1
    def theta(k): return 1 / (np.tan(k * np.pi / 4))
    space_pos = [theta(k / k_max) for k in range(k_max, 0, -1)]
    space_neg = [-i for i in space_pos]
    space_neg.reverse()
    res = np.asarray(space_neg + [0] + space_pos).round(4)
    return res


class IPack(metaclass=ABCMeta):

    @abstractmethod
    def decode(self, decoder):
        pass

    @property
    @abstractmethod
    def node_id(self):
        pass


class QuantizedPack(IPack):

    @staticmethod
    def quantize(arr, encoder, epsilon: float = 1e-9, iterations: int = 3) -> Tuple[float, np.ndarray]:
        """
            Customized function
        :param arr: array to be quantized.
        :param epsilon: epsilon for the denominator of operation.
        :param iterations: iterations for calculation.
        :return: alpha, b.  alpha for the coefficient of quantized array, b for the base
            vector or quantized array.
        """
        a = 0.7
        binaries = encoder.stochastic(arr / a)
        for i in range(iterations):
            b = encoder.decode(binaries)
            a = np.sum(np.multiply(b, arr)) / (np.sum(np.square(b)) + 1)
            binaries = encoder.stochastic(arr / (a + epsilon))
        return a, binaries

    def __init__(self, node_id: int, array: np.ndarray, encoder):
        self.__node_id = node_id
        self.__shape = array.shape
        # float64 only
        array = array.astype('float64')
        self.__alpha, self.__b = QuantizedPack.quantize(array.reshape(-1), encoder)

    def decode(self, decoder):
        return (self.__alpha * decoder.decode(self.__b)).reshape(self.__shape)

    @property
    def node_id(self):
        return self.__node_id


class QuantizedClient(Codec):
    codex = quant(build_quantization_space(int(GlobalSettings.get_params(Quantization_Resolution_Client))))

    def __init__(self, node_id):
        super().__init__(node_id)

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight) -> netEncapsulation[QuantizedPack]:
        package = QuantizedPack(self.node_id, block_weight.content, QuantizedClient.codex)
        return netEncapsulation(Parameter_Server, package)

    def receive_blocks(self, package: IPack):
        self.set_result(package.decode(QuantizedParaServer.codex), lambda x, y: y)


class QuantizedParaServer(Codec):
    codex = quant(build_quantization_space(int(GlobalSettings.get_params(Quantization_Resolution_Server))))

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights: np.ndarray = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, package: IPack):
        self.__global_weights -= package.decode(QuantizedClient.codex)
        reply = QuantizedPack(Parameter_Server, self.__global_weights, QuantizedParaServer.codex)
        return netEncapsulation(package.node_id, reply)
