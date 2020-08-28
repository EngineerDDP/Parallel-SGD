import numpy as np

#custmized module
from utils.huffman import codec

from codec.interfaces import yield_none, netEncapsulation
from codec.interfaces import ICommunication_Ctrl
from codec.essential import Block_Weight
from utils.log import Logger
from utils.constants import Parameter_Server

from profiles.settings import GlobalSettings

# based on paper SGQ chapter 3.1
def build_quantization_space(bits:int) -> list:
    k_max = (2 ** bits - 1) // 2
    theta = lambda k : 1 / (np.tan(k * np.pi / 4))
    space = [theta(k / k_max) for k in range(k_max, 0, -1)]
    return space

q_space_buffer = {2:build_quantization_space(2), 3:build_quantization_space(3)}
q_space_codec_keys = {2: [-1, 0, 1], 3: [-3, -2, -1, 0, 1, 2, 3]}
q_space_codec = {2: codec(), 3: codec()}

q_space_codec[2].set_codec(q_space_codec_keys[2])
q_space_codec[3].set_codec(q_space_codec_keys[3])

def get_quantization_space(bits:int) -> list:
    if q_space_buffer.get(bits, None) is not None:
        return q_space_buffer[bits]
    else:
        q_space_buffer[bits] = build_quantization_space(bits)
        return q_space_buffer[bits]


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
    for i in range(3):
        a = np.sum(np.multiply(b, arr)) / (np.sum(np.square(b)) + 1)
        b = stochastic_ternarization(arr / (a + epsilon))
    return a, b


def stochastic_quantization(arr:np.ndarray, space:list):
    sign = np.sign(arr)
    arr = np.abs(arr)
    for x in np.nditer(arr, op_flags=["readwrite"]):
        lo = 0
        hi = 0
        for i in space:
            if i < x:
                lo = i
            else:
                hi = i
                break

        rnd = np.random.uniform(lo, hi)
        if (rnd > x):
            x = lo
        else:
            x = hi

    return arr

def quantize_matrix(arr):
    raise NotImplementedError()


class SGQClient(ICommunication_Ctrl):

    def __init__(self, node_id, logger=Logger('None')):

        super().__init__()
        self.Node_id = node_id
        self.Logger = logger

    def update_blocks(self, block_weight:Block_Weight):
        """
            Update weights to parameter server
        :param block_weight:
        :return:
        """
        content = block_weight.Content
        pkg = SGQPackage(content, self.Node_id)
        yield netEncapsulation(Parameter_Server, pkg.encode())

    def dispose(self):
        """
            Nothing to do
        """
        pass

    def receive_blocks(self, content):
        """
            Receive a sgq package and set it to result buffer
        :param content:
        :return:
        """
        pkg = SGQPackage.decode(content)
        self.set_result(pkg.content())


class SGQServer(ICommunication_Ctrl):

    __max_error = 0

    def __init__(self, node_id, logger=Logger('None')):

        super().__init__()
        self.Node_Id = node_id
        self.Logger = logger

        self.Global_State = 0
        self.Weights_Last_Received = {}
        for key in GlobalSettings.get_default().nodes:
            self.Weights_Last_Received[key] = 0

    def receive_blocks(self, content):
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
        yield netEncapsulation(data.node_id, data_rtn.encode())


    def update_blocks(self, block_weight):
        """
            SGQ server codec cannot receive data
        """
        return yield_none()

    def dispose(self):
        """
            reset global state and clear all node buffer
        """
        self.Global_State = 0
        self.Weights_Last_Received.clear()


class SGQPackage:

    __var_codec = q_space_codec[2]
    __quant_method = ternarization

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
            self.__alpha, self.__beta = SGQPackage.__quant_method(content)

    def content(self):
        return self.__alpha * self.__beta

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
        res['SGQBETA'] = SGQPackage.__var_codec.encode(self.__beta.astype(int).reshape(-1))
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
        beta = SGQPackage.__var_codec.decode(dic['SGQBETA'])
        beta = beta[:len]
        pkg.__beta = np.asarray(beta, dtype=float).reshape(shape)
        return pkg
