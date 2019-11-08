class ModelMNIST:
    Neural_Network = None

    @staticmethod
    def getWeightsInit():

        if ModelMNIST.Neural_Network is None:
            ModelMNIST.initWeights()

        return ModelMNIST.Neural_Network
    #
    # def initWeights():
    #
    #     from neuralnetworks.layers import FCLayer
    #     from neuralnetworks.activations import Tanh, Sigmoid, ReLU, Linear
    #     from dataset.mnist_input import load_mnist
    #
    #     ServerUtil.Neural_Network = [
    #         FCLayer(1, act=Tanh())
    #     ]
    #
    #     x = ServerUtil.train_data()[0][0]
    #     for layer in ServerUtil.Neural_Network:
    #         x = layer.F(x)

    @staticmethod
    def initWeights():

        from neuralnetworks.layers import FCLayer
        from neuralnetworks.activations import Tanh
        from neuralnetworks.activations import Sigmoid
        from dataset.mnist_input import load_mnist

        ModelMNIST.Neural_Network = [
            FCLayer(256, act=Tanh()),
            FCLayer(128, act=Tanh()),
            FCLayer(128, act=Tanh()),
            FCLayer(128, act=Tanh()),
            FCLayer(128, act=Tanh()),
            FCLayer(10, act=Sigmoid())]

        input_sample = load_mnist()[0][0].reshape([-1, 1])
        for layer in ModelMNIST.Neural_Network:
            input_sample = layer.F(input_sample)

        # ModelMNIST.Neural_Network = [(i.Output, i.W, i.B, i.Act) for i in ModelMNIST.Neural_Network]
        # ServerUtil.Neural_Network = np.linspace(0,1,100).reshape([10,10])
        return

    @staticmethod
    def codec_ctrl():

        from codec.ccdc import CodedCommunicationCtrl
        from codec.plain import PlainCommunicationCtrl
        from codec.pacodec import PAClientCodec

        return CodedCommunicationCtrl

    @staticmethod
    def psgd_type():

        from psgd.asgd import AsynchronizedSGD
        from psgd.ssgd import SynchronizedSGD

        return SynchronizedSGD

    @staticmethod
    def loss_type():

        from neuralnetworks.losses import CrossEntropyLossWithSigmoid
        from neuralnetworks.losses import MseLoss
        from neuralnetworks.losses import CrossEntropyLoss

        return CrossEntropyLossWithSigmoid

    @staticmethod
    def epoches():

        return 100

    @staticmethod
    def learn_rate():

        return 0.05

    T_DATA = None
    #
    # @staticmethod
    # def train_data():
    #     from dataset.mnist_input import load_mnist
    #     from dataset.simdata import load_lin_sim, load_sin_sim
    #
    #     if ServerUtil.T_DATA is None:
    #         ServerUtil.T_DATA = load_sin_sim()
    #         return ServerUtil.T_DATA
    #     else:
    #         return ServerUtil.T_DATA
    #
    # E_DATA = None
    #
    # @staticmethod
    # def eval_data():
    #     from dataset.mnist_input import load_mnist
    #     from dataset.simdata import load_lin_sim, load_sin_sim
    #
    #     if ServerUtil.E_DATA is None:
    #         ServerUtil.E_DATA = load_sin_sim()
    #         return ServerUtil.E_DATA
    #     else:
    #         return ServerUtil.E_DATA

    @staticmethod
    def train_data():
        from dataset.mnist_input import load_mnist

        return load_mnist(kind='train')

    @staticmethod
    def eval_data():
        from dataset.mnist_input import load_mnist

        return load_mnist(kind='t10k')