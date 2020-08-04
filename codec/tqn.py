import numpy as np


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

    def __init__(self, content, node_id = -2):
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
        res['TQNBETA'] = self.__beta
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
        pkg.__beta = dic['TQNBETA']
        return pkg

