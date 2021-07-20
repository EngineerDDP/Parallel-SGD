import numpy as np

from parallel_sgd.codec.essential import BlockWeight
from parallel_sgd.codec.interfaces import Codec, netEncapsulation
from constants import Parameter_Server


def stochastic_ternarization(arr):
    """
        Stochastic quantization implementation based on paper in NIPS:
        Alistarh et al. “QSGD: Communication-Efficient SGD via Gradient
        Quantization and Encoding”. NIPS2017
    """
    return np.asarray((np.random.random(arr.shape) < np.abs(arr)) * np.sign(arr))


def ternarization(arr, epsilon=1e-9):
    """
        TERNARIZATION TQN implementation based on chapter 3.1.1 in
        L. Hou and J. T. Kwok. Loss-aware weight quantization of deep networks.
        In International Conference on Learning Representations (ICLR), 2018.
    """
    a = 0.7
    b = stochastic_ternarization(arr / a)
    for i in range(1):
        a = np.sum(np.multiply(b, arr)) / (np.sum(np.square(b)) + 1)
        b = stochastic_ternarization(arr / (a + epsilon))
    return a, b


class TQNPackage:

    def __init__(self, content, node_id=-2):
        """
            Build SGQ transmitting package
        :param content: weights delta content
        :param node_id: node_id where the package from
        """
        self.node_id = node_id
        self.__alpha = 0
        self.__beta = 0
        if content is not None:
            self.__alpha, self.__beta = ternarization(content)

    def content(self):
        return self.__alpha * self.__beta

    def encode(self):
        """
            return encode object for network transmission
            Class must have decode process in pair
        :return: encode object
        """
        res = dict()
        res['TQNNode_ID'] = self.node_id
        res['TQNALPHA'] = self.__alpha
        res['TQNBETA'] = self.__beta.astype('int8')
        return res

    @staticmethod
    def decode(dic):
        """
            decode process cooperate with obj.encode()
        :param dic: the result from obj.encode()
        :return: rebuild object
        """
        pkg = TQNPackage(None, dic['TQNNode_ID'])
        pkg.__alpha = dic['TQNALPHA']
        pkg.__beta = dic['TQNBETA'].astype('float64')
        return pkg


class TQNClient(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)

    def update_blocks(self, block_weight: BlockWeight):
        return netEncapsulation(Parameter_Server, TQNPackage(block_weight.content, self.node_id).encode())

    def dispose(self):
        pass

    def receive_blocks(self, content: dict):
        pkg = TQNPackage.decode(content)
        self.set_result(pkg.content())


class TQNServer(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights = 0

    def update_blocks(self, block_weight: BlockWeight):
        pass

    def receive_blocks(self, content: dict):
        pkg = TQNPackage.decode(content)
        self.__global_weights -= pkg.content()
        return netEncapsulation(pkg.node_id, TQNPackage(self.__global_weights, Parameter_Server).encode())

    def dispose(self):
        pass
