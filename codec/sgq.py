import numpy as np

from codec.interfaces import yield_none, NetEncapsulation
from codec.interfaces import ICommunicationCtrl
from codec.essential import BlockWeight
from log import Logger
from network.agreements import DefaultNodes

from profiles.settings import GlobalSettings

# based on paper SGQ chapter 3.1
def build_quantization_space(bits):
    vals = (2 ^ bits - 1) // 2
    theta = lambda k : 1 / (np.tan(k * np.pi / 4))
    space = [theta(k) for k in range(-vals, vals, 1)]
    return space


def stochastic_ternarization(arr):
    """
        Stochastic quantization implementation based on paper in NIPS:
        Alistarh et al. “QSGD: Communication-Efficient SGD via Gradient
        Quantization and Encoding”. NIPS2017
    """
    return np.asarray((np.random.random(arr.shape) < np.abs(arr)) * np.sign(arr), np.int8)


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


def stochastic_quantization(arr, space):
    raise NotImplementedError()


def quantize_matrix(arr):
    raise NotImplementedError()


class SGQClient(ICommunicationCtrl):

    def __init__(self, node_id, logger=Logger('None')):

        super().__init__()
        self.Node_id = node_id
        self.Logger = logger

    def update_blocks(self, block_weight:BlockWeight):
        """
            Update weights to parameter server
        :param block_weight:
        :return:
        """
        content = block_weight.Content
        pkg = SGQPackage(content, self.Node_id)
        yield NetEncapsulation(DefaultNodes.Parameter_Server, pkg.encode())

    def dispose(self):
        """
            Nothing to do
        """
        pass

    def receive_blocks(self, json_dict):
        """
            Receive a sgq package and set it to result buffer
        :param json_dict:
        :return:
        """
        pkg = SGQPackage.decode(json_dict)
        self.set_result(pkg.content())


class SGQServer(ICommunicationCtrl):

    __max_error = 0

    def __init__(self, node_id, logger=Logger('None')):

        super().__init__()
        self.Node_Id = node_id
        self.Logger = logger

        self.Global_State = 0
        self.Weights_Last_Received = {}
        for key in GlobalSettings.get_default().nodes:
            self.Weights_Last_Received[key] = 0

    def receive_blocks(self, json_dict):
        """
            SGQ receive a quantized matrix and returns the grad diff for the client
        :param json_dict:
        :return:
        """
        # decode data
        data = SGQPackage.decode(json_dict)
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
            print(error)
        # return
        yield NetEncapsulation(data.node_id, data_rtn.encode())


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
        res['SGQNode_ID'] = self.node_id
        res['SGQALPHA'] = self.__alpha
        res['SGQBETA'] = self.__beta
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
        pkg.__beta = dic['SGQBETA']
        return pkg
