from abc import ABCMeta, abstractmethod


class IOptimizer(metaclass=ABCMeta):

    @abstractmethod
    def optimize(self, layers):
        """
            Set layers to be optimized.
            layers has its own properties to describe that it can be optimized or not.
        :param layers: Full network layers
        :return: None
        """
        pass

    @abstractmethod
    def set_loss(self, loss_function):
        """
            Set loss function.
            This loss function will be used to calculate gradients.
        :param loss_function: ILoss type
        :return: None
        """
        pass

    @abstractmethod
    def train(self, x, y):
        """
            Train the provided layer.
        :param x: input samples
        :param y: label of samples
        :return: None
        """
        pass


class IActivation(metaclass=ABCMeta):

    @abstractmethod
    def activation(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass


class ILayer(metaclass=ABCMeta):

    def __init__(self):
        """
            Lazy initialization
        """
        # redirect function calls
        self.logit = self.logit_with_init_parameters

    @abstractmethod
    def param(self):
        """
            Essential parameters for reconstruction.
        :return: tuple
        """
        pass

    @property
    @abstractmethod
    def Variables(self):
        """
            Get variables from this layer
        :return: tuple that contains all the variables
        """
        pass

    @property
    @abstractmethod
    def Input(self):
        """
            Describe the input shape of this layer.
        :return: Shape object, tuple or int.
        """
        pass

    @property
    @abstractmethod
    def Output(self):
        """
            Describe the output shape of this layer.
        :return: Shape object, tuple or int.
        """
        pass

    @property
    @abstractmethod
    def Trainable(self):
        """
        return if this layer can be trained.
        :return: [True/false]
        """
        pass

    def __trainable_check(self, func_name):
        """
            If this layer is not trainable, then pass the check.
        :return:
        """
        if self.Trainable:
            raise NotImplementedError('abstract method "{}" not implemented.'.format(func_name))
        else:
            pass

    def reset(self):
        """
            Reset this layer, all parameters in this layer will be reinitialized.
            Hyper parameters remains unchanged.
        :return: None
        """
        self.logit = self.logit_with_init_parameters
        return self.clear_parameters()

    def clear_parameters(self):
        """
            Clear all existed parameters.
        :return: None
        """
        self.__trainable_check('ILayer.clear_parameters()')

    def initialize_parameters(self, x):
        """
            Initialize all parameters in this layer.
        :return: None
        """
        self.__trainable_check('ILayer.initialize_parameters(x)')

    def logit_with_init_parameters(self, x):
        """
            Calculate logit result, and redirect function calls
            code reconstructed by Chen R.
        :param x: input batch or single sample
        :return: z output, without activation
        """
        self.initialize_parameters(x)
        self.logit = self.calculate_logit
        return self.calculate_logit(x)

    @abstractmethod
    def calculate_logit(self, x):
        """
            Calculate logit.
            y = x * w + b / y = w * x + b
        """
        pass

    @abstractmethod
    def F(self, x):
        """
            output function
        """
        pass

    @abstractmethod
    def delta_wb(self, x, gradient):
        pass

    @abstractmethod
    def apply_wb(self, w, b, y):
        pass

    @abstractmethod
    def backpropagation(self, x, gradient):
        """
            Calculate gradient, adjust weight and bias and return gradients of this layer
            x shape=[input, samples count]
            grad shape=[output, samples count]
        """
        pass


class IMetrics(metaclass=ABCMeta):

    @abstractmethod
    def metric(self, y, label):
        """
            Calculate metrics value
        :return: Scala: single metrics value
        """
        pass

    @abstractmethod
    def description(self):
        """
            Official name for this metric.
        :return:
        """
        pass


class ILoss(IMetrics):
    """
        General loss function interface.
        All loss function implements IMetrics
    """

    @abstractmethod
    def gradient(self, y, label):
        """
            Calculate gradient for backward propagation.
        :param y: prediction
        :param label: label
        :return: vector result
        """
        pass

    def description(self):
        return 'loss'
