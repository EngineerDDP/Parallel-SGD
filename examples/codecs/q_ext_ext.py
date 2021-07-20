from abc import ABCMeta, abstractmethod
from typing import Sequence, Tuple, Dict

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


def build_quantization_space(bits: int) -> Sequence[float]:
    """
        Build quantization space
    :param bits: bits of input
    :return: list of float
    """
    states = max(bits, 2)
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
        # Use int8 for fast serialization.
        self.__alpha, self.__b = QuantizedPack.quantize(array.reshape(-1), encoder)

    def decode(self, decoder):
        return (self.__alpha * decoder.decode(self.__b)).reshape(self.__shape)

    @property
    def node_id(self):
        return self.__node_id


class SignalPack(IPack):

    def __init__(self, node_id):
        self.__id = node_id

    def decode(self, decoder):
        return 0

    @property
    def node_id(self):
        return self.__id


class DC_QSGDClient(Codec):
    codex = quant(build_quantization_space(int(GlobalSettings.get_params(Quantization_Resolution_Client))))

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__gw_t: np.ndarray = 0
        self.__lambda = 0.3

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight) -> netEncapsulation[IPack]:
        self.__gw_t = block_weight.content
        return netEncapsulation(Parameter_Server, SignalPack(self.node_id))

    def receive_blocks(self, package: IPack) -> netEncapsulation[IPack]:
        delta_w = package.decode(DC_QSGDServer.codex)
        ggw_t = np.multiply(np.multiply(self.__gw_t, self.__gw_t), delta_w)
        gw_t_tau = self.__gw_t + self.__lambda * ggw_t
        self.set_result(gw_t_tau - delta_w, lambda x, y: x + y if x is not None else y)
        return netEncapsulation(Parameter_Server, QuantizedPack(self.node_id, gw_t_tau, DC_QSGDClient.codex))


class DC_QSGDServer(Codec):
    codex = quant(build_quantization_space(int(GlobalSettings.get_params(Quantization_Resolution_Server))))

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights: np.ndarray = 0
        self.__weights_states: Dict[int, np.ndarray] = {}

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, package: IPack):
        if isinstance(package, QuantizedPack):
            self.__global_weights -= package.decode(DC_QSGDClient.codex)
            self.__weights_states[package.node_id] = self.__global_weights
        elif isinstance(package, SignalPack):
            # returns Q(w_t+\tau - w_t)
            if not isinstance(self.__global_weights, np.ndarray):
                reply = SignalPack(Parameter_Server)
            else:
                reply = QuantizedPack(Parameter_Server,
                                      self.__global_weights - self.__weights_states.get(package.node_id, 0),
                                      DC_QSGDServer.codex)
            return netEncapsulation(package.node_id, reply)
