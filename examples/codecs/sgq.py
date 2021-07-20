import numpy as np
# customized module
from utils.huffman import codec

from parallel_sgd.codec import GlobalSettings
from parallel_sgd.codec.essential import BlockWeight
from parallel_sgd.codec.interfaces import Codec
from parallel_sgd.codec.interfaces import netEncapsulation
from constants import Parameter_Server

"""
    This class has been deprecated.
    Quantization with huffman encoding has much better implementations than this.
"""

"""
    This codec requires specified parameters.
    Listed as below:
"""

Quantization_Resolution_Client = "QC"

"""
    Parameters listed above should be added to GlobalSettings.global_parameters as dict type.
    Fill the parameter "codec_extra_parameters" while calling rpc.parallel_sgd.submit.ParallelSGD.parallel_computing()
    with this codec.
"""


# based on paper SGQ chapter 3.1
def build_quantization_space(states: int) -> list:
    k_max = states // 2
    def theta(k): return 1 / (np.tan(k * np.pi / 4))
    space_pos = [theta(k / k_max) for k in range(k_max, 0, -1)]
    space_neg = [-i for i in space_pos]
    space_neg.reverse()
    res = np.asarray(space_neg + [0] + space_pos).round(4)
    return res


def stochastic_ternarization(arr):
    """
        Stochastic quantization implementation based on paper in NIPS:
        Alistarh et al. “QSGD: Communication-Efficient SGD via Gradient
        Quantization and Encoding”. NIPS2017
    """
    return np.asarray((np.random.random(arr.shape) < np.abs(arr)) * np.sign(arr))


def quantize(arr, space, epsilon: float = 1e-9, iterations: int = 3):
    """
        TERNARIZATION TQN implementation based on chapter 3.1.1 in
        L. Hou and J. T. Kwok. Loss-aware weight quantization of deep networks.
        In International Conference on Learning Representations (ICLR), 2018.
    """
    a = 0.7
    b = stochastic_quantization(arr / a, space)
    for i in range(iterations):
        a = np.sum(np.multiply(b, arr)) / (np.sum(np.square(b)) + 1)
        b = stochastic_quantization(arr / (a + epsilon), space)
    return a, b


def stochastic_quantization(arr: np.ndarray, space: list):
    if len(space) == 3:
        return stochastic_ternarization(arr)
    sign = np.sign(arr)
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

    return sign * arr_in


class SGQClient(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)

    def update_blocks(self, block_weight: BlockWeight):
        """
            Update weights to parameter server
        :param block_weight:
        :return:
        """
        content = block_weight.content
        pkg = SGQPackage(content, self.node_id)
        return netEncapsulation(Parameter_Server, pkg.encode())

    def dispose(self):
        """
            Nothing to do
        """
        pass

    def receive_blocks(self, content: dict):
        """
            Receive a sgq package and set it to result buffer
        :param content:
        :return:
        """
        pkg = SGQPackage.decode(content)
        self.set_result(pkg.content())


class SGQServer(Codec):
    __max_error = 0

    def __init__(self, node_id):

        super().__init__(node_id)

        self.Global_State = 0
        self.Weights_Last_Received = {}
        for key in GlobalSettings.get_default().nodes:
            self.Weights_Last_Received[key] = 0

    def receive_blocks(self, content: dict):
        """
            SGQ receive a quantized matrix and returns the grad diff for the client
        :param content:
        :return:
        """
        # decode data
        data = SGQPackage.decode(content)
        # get working node state
        last_state = self.Weights_Last_Received[data.node_id]
        # update global state
        self.Global_State = self.Global_State + data.content()
        # get grad diff
        grad_diff = self.Global_State - last_state
        # encode data
        data_rtn = SGQPackage(grad_diff)
        # update state buffer
        self.Weights_Last_Received[data.node_id] = last_state + data_rtn.content()
        # for DEBUG
        error = np.sum(self.Weights_Last_Received[data.node_id] - self.Global_State)
        if error > SGQServer.__max_error:
            SGQServer.__max_error = error
            print("Error:", error)
        # return
        return netEncapsulation(data.node_id, data_rtn.encode())

    def update_blocks(self, block_weight):
        """
            SGQ server codec cannot receive data
        """
        pass

    def dispose(self):
        """
            reset global state and clear all node buffer
        """
        self.Global_State = 0
        self.Weights_Last_Received.clear()


class SGQPackage:
    __quant_codec = None
    __quant_code = None
    __quant_space = None

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
            self.__alpha, self.__beta = quantize(content, SGQPackage.__quant_space)

    def content(self):
        return self.__alpha * self.__beta

    @staticmethod
    def build():
        SGQPackage.__quant_codec = codec()
        SGQPackage.__quant_space = build_quantization_space(
            2 ** int(GlobalSettings.get_params(Quantization_Resolution_Client)) - 1)
        SGQPackage.__quant_code = []
        i = -1

        while len(SGQPackage.__quant_code) != len(SGQPackage.__quant_space):
            i += 1
            flag = True
            for c in SGQPackage.__quant_space:
                if abs(c - i) < 1e-3:
                    flag = False
                    break
            if flag:
                SGQPackage.__quant_code.append(i)
        SGQPackage.__quant_codec.set_codec(SGQPackage.__quant_code)

    @staticmethod
    def __encode(arr):
        it = iter(SGQPackage.__quant_code)
        for v in SGQPackage.__quant_space:
            arr[arr == v] = next(it)
        return SGQPackage.__quant_codec.encode(arr.reshape(-1).astype(int))

    @staticmethod
    def __decode(_bytes, _len):
        arr = np.asarray(SGQPackage.__quant_codec.decode(_bytes)[:_len])
        it = iter(SGQPackage.__quant_code)
        for v in SGQPackage.__quant_space:
            arr[arr == next(it)] = v
        return arr

    def encode(self):
        """
            return encode object for network transmission
            Class must have decode process in pair
        :return: encode object
        """
        res = dict()
        res['SGQNode_ID'] = self.node_id
        res['SGQALPHA'] = self.__alpha
        res['SGQBETA_SHAPE'] = self.__beta.shape
        res['SGQBETA'] = SGQPackage.__encode(self.__beta)
        return res

    @staticmethod
    def decode(dic):
        """
            decode process cooperate with obj.encode()
        :param dic: the result from obj.encode()
        :return: rebuild object
        """
        pkg = SGQPackage(None, dic['SGQNode_ID'])
        pkg.__alpha = dic['SGQALPHA']
        # get shape
        len = 1
        shape = dic['SGQBETA_SHAPE']
        for i in shape:
            len *= i
        # rebuild beta
        beta = SGQPackage.__decode(dic['SGQBETA'], len)
        pkg.__beta = np.asarray(beta, dtype=float).reshape(shape)
        return pkg


SGQPackage.build()

if __name__ == '__main__':
    rand = np.random.uniform(low=-1, high=1, size=[784, 784])
    a = SGQPackage(rand)
    source = a.content()
    target = SGQPackage.decode(a.encode()).content()

    print(np.abs(rand - target).sum())
