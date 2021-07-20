from abc import ABCMeta, abstractmethod
from typing import Sequence, Tuple

import numpy as np

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


def build_quantization_space(bits: int = 2) -> Sequence[int]:
    bits = max(bits, 2)
    max_v = (1 << bits - 1) - 1
    min_v = -max_v
    return np.arange(min_v, max_v + 1, 1)


def deterministic_quantization(arr: np.ndarray, space: Sequence[int]):
    arr[arr > space[-1]] = space[-1]
    arr[arr < space[0]] = space[0]
    return np.round(arr)


def stochastic_quantization(arr: np.ndarray, space: Sequence[int]):
    """
        Alistarh et al. “QSGD: Communication-Efficient SGD via Gradient Quantization
        and Encoding”. NIPS2017
    :param arr: array to be quantized.
    :param space: quantization space.
    :return: the base vector or quantized array.
    """
    arr[arr > space[-1]] = space[-1]
    arr[arr < space[0]] = space[0]
    _floor = np.floor(arr)
    _rand = np.random.uniform(low=0.0, high=1.0, size=_floor.shape)

    return (_floor + _rand < arr).astype('int') + _floor


def Q_g(arr: np.ndarray, space: Sequence[int], epsilon: float = 1e-9, iterations: int = 3) -> Tuple[float, np.ndarray]:
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
    b = stochastic_quantization(arr / a, space)
    for i in range(iterations):
        a = np.sum(np.multiply(b, arr)) / (np.sum(np.square(b)) + 1)
        b = stochastic_quantization(arr / (a + epsilon), space)
    return a, b


def Q_w(w: np.ndarray, s: Sequence[int], d: np.ndarray, epsilon: float = 1e-4) -> Tuple[float, np.ndarray]:
    """
        weights quantization method
        L. Hou and J. T. Kwok. Loss-aware weight quantization of deep networks.
        In International Conference on Learning Representations (ICLR), 2018.
    :param w: weights
    :param s: quantization space
    :param d: estimation of hessian matrix
    :param epsilon:
    :return: quantized alpha and matrix
    """
    a = 1.0
    a_old = 0.0
    b = stochastic_quantization(w, s)
    while abs(a - a_old) > epsilon:
        a_old = a
        a = np.sum(np.multiply(np.multiply(b, w), d)) / np.sum(np.multiply(np.square(b), d))
        b = stochastic_quantization(w / (a + 1e-7), s)
    return a, b


def binarize(arr: np.ndarray) -> Tuple[float, np.ndarray]:
    """
        M. Courbariaux, Y. Bengio, and J. P. David. BinaryConnect:
        Training deep neural networks with binary weights during propagations.
        In Advances in Neural Information Processing Systems,
        pp.3105–3113, 2015.
    :param arr: arr to be binarized.
    :return: alpha, b.  alpha for the coefficient of quantized array, b for the base
        vector or quantized array.
    """
    alpha: float = np.std(arr)[()]
    b = np.sign(arr)
    return alpha, b


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

    def __init__(self, node_id: int, alpha: float, b: np.ndarray):
        self.__node_id = node_id
        self.__alpha = alpha
        # Use int8 for fast serialization.
        self.__b = b.astype('int8')

    @property
    def content(self):
        return self.__alpha * self.__b.astype('float64')

    @property
    def node_id(self):
        return self.__node_id


class QuantizedClient(Codec):
    q_space = build_quantization_space(int(GlobalSettings.get_params(Quantization_Resolution_Client)))

    def __init__(self, node_id):
        super().__init__(node_id)

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight) -> netEncapsulation[QuantizedPack]:
        package = QuantizedPack(self.node_id, *Q_g(block_weight.content, QuantizedClient.q_space))
        return netEncapsulation(Parameter_Server, package)

    def receive_blocks(self, package: IPack):
        self.set_result(package.content, lambda x, y: y)


class QuantizedParaServer(Codec):
    q_space = build_quantization_space(int(GlobalSettings.get_params(Quantization_Resolution_Server)))

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights: np.ndarray = 0
        self.__vt: np.ndarray = 0
        self.__beta: float = 0.9

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, package: IPack):
        grad = package.content
        self.__global_weights -= grad
        self.__vt = self.__beta * self.__vt + np.square(grad) * (1 - self.__beta)
        reply = QuantizedPack(Parameter_Server,
                              *Q_w(self.__global_weights, QuantizedParaServer.q_space, np.sqrt(self.__vt)))
        return netEncapsulation(package.node_id, reply)
