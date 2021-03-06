from abc import ABCMeta, abstractmethod
from typing import Sequence, Tuple

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
    arr[arr > space[-1]] = space[-1]
    arr[arr < space[0]] = space[0]
    _floor = np.floor(arr)
    _rand = np.random.uniform(low=0.0, high=1.0, size=_floor.shape)

    return (_floor + _rand < arr).astype('int') + _floor


def quantize(arr, space, epsilon: float = 1e-9, iterations: int = 3) -> Tuple[float, np.ndarray]:
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


class NormalPack(IPack):

    def __init__(self, node_id: int, data: np.ndarray):
        self.__content = data
        self.__node_id = node_id

    @property
    def content(self):
        return self.__content

    @property
    def node_id(self):
        return self.__node_id


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


class BinaryNetCodec(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.epsilon = 0.01

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight) -> netEncapsulation[QuantizedPack]:
        package = QuantizedPack(self.node_id, *binarize(block_weight.content))
        return netEncapsulation(Parameter_Server, package)

    def receive_blocks(self, package: IPack):
        self.set_result(package.content, lambda x, y: y)


class QuantizedClient(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight) -> netEncapsulation[QuantizedPack]:
        package = QuantizedPack(self.node_id, *quantize(block_weight.content, q_space))
        return netEncapsulation(Parameter_Server, package)

    def receive_blocks(self, package: IPack):
        self.set_result(package.content, lambda x, y: y)


class FullPrecisionParaServer(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights: np.ndarray = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, package: IPack):
        self.__global_weights -= package.content
        reply = NormalPack(Parameter_Server, self.__global_weights)
        return netEncapsulation(package.node_id, reply)


class LowPrecisionParaServer(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights: np.ndarray = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, package: IPack):
        self.__global_weights -= package.content
        reply = NormalPack(Parameter_Server, self.__global_weights.astype('float16'))
        return netEncapsulation(package.node_id, reply)


class QuantizedParaServer(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights: np.ndarray = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, package: IPack):
        self.__global_weights -= package.content
        reply = QuantizedPack(Parameter_Server, *quantize(self.__global_weights, q_space))
        return netEncapsulation(package.node_id, reply)


class BinaryParaServer(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights: np.ndarray = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, package: IPack):
        self.__global_weights -= package.content
        reply = QuantizedPack(Parameter_Server, *binarize(self.__global_weights))
        return netEncapsulation(package.node_id, reply)
