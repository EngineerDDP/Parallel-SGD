from abc import ABCMeta, abstractmethod


class IDispatcher(ABCMeta):

    def __init__(self, sgds, com):
        super.__init__(self)
        self.SGDS = sgds
        self.Com = com

    @abstractmethod
    def send_out(self, send_to, obj):
        pass


class IParallelSGD:
    """
        Working in one specified layer.
    """

    def __init__(self, node_id, layer_id, codec, code='0'):
        self.Node_ID = node_id
        self.Layer_ID = layer_id
        self.Updater = codec
        self.Code = code

    def init_startup_setting(self, params=None):
        """
            Setup for sgd job.
        :param params: initialization parameters
        :return: None
        """
        pass

    def release_memory(self):
        """
            Release memory used by this sgd controller.
        :return:
        """
        pass

    def update_weights(self, content, tag):
        """
            Update a calculated weights to this controller.
            Weights may be calculated throw partial samples.
        :return: json object to be sent throw network,
                 None if nothing needs to be sent.
        """
        pass

    def require_weights(self, tag):
        """
            Require a aggregated full calculated newest weights.
        :return: Weights Matrix: Numpy Array
        """
        pass

    def accept_data(self, obj):
        """
            Receive a decomposable object for local weights update.
            object received from local dispatcher
        :return: json object to be sent throw network,
                 None if nothing needs to be sent.
        """
        pass


class ITransfer(metaclass=ABCMeta):

    @abstractmethod
    def put_weights(self, content, tag, w_type='w'):
        """
            Put a intermediate weights.
            distributed to specified layer sgd processor (IParallelSGD).
        :param content: content of the weights : ndarray
        :param tag: description of the weights : codec.tag.Tag
        :param w_type: type of weights : str
        :return: None
        """
        pass

    @abstractmethod
    def get_weights(self, tag, w_type='w'):
        """
            Acquire intermediate weights from local layer sgd processor (IParallelSGD)
        :param tag: description of the weights : codec.tag.Tag
        :param w_type: type of the weights : str
        :return: content of the weights : ndarray
        """
        pass

    @abstractmethod
    def start_transfer(self):
        """
            Start transferring data between working process and
            network communication process.
        :return: None
        """
        pass