class ServerUtil:
    Neural_Network = None

    @staticmethod
    def getWeightsInit():

        if ServerUtil.Neural_Network is None:
            ServerUtil.initWeights()

        return ServerUtil.Neural_Network

    @staticmethod
    def initWeights():

        from neuralnetworks.layers import FCLayer
        from neuralnetworks.activations import Tanh
        from neuralnetworks.activations import Sigmoid
        from dataset.mnist_input import load_mnist

        ServerUtil.Neural_Network = [
            FCLayer(256, act=Tanh()),
            FCLayer(128, act=Tanh()),
            FCLayer(128, act=Tanh()),
            FCLayer(128, act=Tanh()),
            FCLayer(128, act=Tanh()),
            FCLayer(10, act=Sigmoid())]

        input_sample = load_mnist()[0][0].reshape([-1, 1])
        for layer in ServerUtil.Neural_Network:
            input_sample = layer.F(input_sample)

        ServerUtil.Neural_Network = [(i.Output, i.W, i.B, i.Act) for i in ServerUtil.Neural_Network]
        # ServerUtil.Neural_Network = np.linspace(0,1,100).reshape([10,10])
        return

    @staticmethod
    def codec_ctrl():

        from codec.ccdc import CodedCommunicationCtrl
        from codec.plain import PlainCommunicationCtrl

        return PlainCommunicationCtrl

    @staticmethod
    def psgd_type():

        from psgd.asgd import ASGD
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
