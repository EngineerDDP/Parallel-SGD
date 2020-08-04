import numpy as np

from nn.interfaces import ILayer

from nn.activations import Linear
from nn.activations import ReLU


class FCLayer_v2(ILayer):
    """
        Full connected layer version 2.0, compatible with convolution layers.
    """

    def __init__(self, units, w_init=None, b_init=None, act=Linear()):

        # use lazy initialization
        super().__init__()
        if w_init is not None:
            self.W = w_init
        else:
            self.W = None

        if b_init is not None:
            self.B = b_init
        else:
            self.B = None

        self.Act = act
        self.__output = units

    @property
    def Trainable(self):
        return True

    def clear_parameters(self):
        self.W = None
        self.B = None

    @property
    def Output(self):
        return self.__output

    @property
    def Input(self):
        return 'unknown' if self.W is None else self.W.shape[0]

    def calculate_logit(self, x):
        return np.dot(x, self.W) + self.B

    def initialize_parameters(self, x):
        if self.W is None:
            high = np.sqrt(6 / (x.shape[1] + self.Output))
            low = -high
            self.W = np.random.uniform(low=low, high=high, size=[x.shape[1], self.Output])
        if self.B is None:
            self.B = np.zeros(shape=[1, self.Output])

        return None

    def F(self, x):
        """
            output function
        """
        # activation
        return self.Act.activation(self.logit(x))

    def delta_wb(self, x, gradient):
        # calculate gradient
        act_grad = self.Act.gradient(self.logit(x))
        # y shape=[output, samples count]
        y_grad = np.multiply(act_grad, gradient)

        # adjust weight
        delta_weight = (y_grad.T.dot(x)).T
        delta_bias = y_grad.sum(axis=0)

        return delta_weight, delta_bias, y_grad

    def apply_wb(self, w, b, y):

        self.W = self.W - w
        self.B = self.B - b
        # calculate gradient for BP
        grad = y.dot(self.W.transpose())
        return grad

    def forward_gradient(self, x, gradient):
        # calculate gradient
        act_grad = self.Act.gradient(self.logit(x))
        # y shape=[output, samples count]
        y = np.multiply(act_grad, gradient)
        # calculate gradient for BP
        grad = y.dot(self.W.transpose())
        return grad

    def backpropagation(self, x, gradient):
        """
            Calculate gradient, adjust weight and bias and return gradients of this layer
            x shape=[input, samples count]
            grad shape=[output, samples count]
        """
        w, b, y = self.delta_wb(x, gradient)
        w = w / x.shape[0]
        b = b / x.shape[0]
        return self.apply_wb(w, b, y)


class Conv2dLayer(ILayer):

    def __init__(self, filter_size, channel_count, padding, strikes, act=ReLU()):
        """
            Initialize a 2-dimensional convolution layer.
            Input format:
                4D tensor, shape like [sample, height, width, channels]
                eg.
                [None, 128, 128, 3]
        :param filter_size: Shape of filter. [lines, columns] :list[int]
        :param channel_count: Number of filter :int
        :param padding: Convolution padding for feature map. [left, up, right, bottom] :list[int]
        :param strikes: Step per operation. [rows skip, columns skip] :list[int]
        """
        # Convolution OP parameters
        super().__init__()
        if padding == 'SAME':
            self.Padding = [0, 0, 0, 0]
        else:
            self.Padding = padding
        self.Strikes = strikes
        # Convolution Kernel parameters
        self.Filter_Size = filter_size
        self.Kernel_Count = channel_count
        # Activation function
        self.Act = act
        # Kernels
        self.Kernels = None
        self.Bias = None
        self.Previous_Channels = None

    def __w_init_xvars(self, shape):
        high = np.sqrt(6 / (shape[0] + shape[1]))
        low = -high
        return np.random.uniform(low=low, high=high, size=shape)

    def __init_kernel(self, w_init=None):

        if w_init is not None:
            self.Kernels = [w_init[i] for i in range(self.Kernel_Count)]
        else:
            shape = [self.Filter_Size[0], self.Filter_Size[1], self.Previous_Channels]
            self.Kernels = np.asarray([self.__w_init_xvars(shape) for i in range(self.Kernel_Count)])

    def __init_bias(self, b_init=None):

        if b_init is not None:
            self.Bias = b_init
        else:
            self.Bias = np.zeros(shape=[self.Kernel_Count])

    def __conv2d(self, x, y, padding=None, strikes=None):
        """
            Convolution between x and y
        """
        # using default
        if padding is None:
            padding = self.Padding
        if strikes is None:
            strikes = self.Strikes
        # Calculate shape of the result
        result_shape = [x.shape[0] - (padding[1] + padding[3]), x.shape[1] - (padding[0] + padding[2])]
        # calculated shape of expanded x
        x_exp = np.zeros(shape=[y.shape[0] + result_shape[0] - 1, y.shape[1] + result_shape[1] - 1])
        # calculate position of real x in expanded x
        x_exp_inner_start = [y.shape[0] // 2 - padding[1], y.shape[1] // 2 - padding[0]]
        # write real x to expanded x
        x_exp[x_exp_inner_start[0]:x_exp_inner_start[0] + x.shape[0], x_exp_inner_start[1]:x_exp_inner_start[1] + x.shape[1]] = x
        # partial convolution helper
        conv_2d = lambda i, j:np.sum(np.multiply(x_exp[i:y.shape[0]+i, j:y.shape[1]+j], y))
        # build result
        result = np.asarray([[conv_2d(i, j) for j in range(0, result_shape[1], strikes[1])] for i in range(0, result_shape[0], strikes[0])])

        return result

    def __channel_conv(self, feature_map, kernel):
        """
            Calculate multi-channel convolution
            Feature map is 3-dimensional [Height, Width, Channels]
            Kernel is 3-dimensional [Height, Width, Channels]
        """

        # conv for each dimension
        conv = np.sum([self.__conv2d(feature_map[:,:,i], kernel[:,:,i]) for i in range(self.Previous_Channels)], axis=0)
        return conv

    def __channel_conv_revt(self, x, y):
        """
            Calculated convolution result
            convolution of single x and multiple ys.
            with all y[:,:,i] matrix rot 180 degrees
        """
        # get 3-d result
        conv = np.asarray([self.__conv2d(x, np.rot90(y[:,:,i], 2)) for i in range(self.Previous_Channels)])
        # reformat dimensions
        conv = np.swapaxes(np.swapaxes(conv, 1, 3), 1, 2)
        return conv

    def __channel_conv_upsample(self, feature_map, kernel):
        """
            Calculate deconvolution result
            with x like: [height, width, out_channel]
            with y like: [out_channel, height, width]
        :param feature_map:
        :param kernel:
        :return:
        """
        conv = np.sum([self.__conv2d(feature_map[:,:,i], np.rot90(kernel[i,:,:], 2)) for i in range(self.Kernel_Count)], axis=0)
        return conv

    @property
    def Trainable(self):
        return True

    @property
    def Output(self):
        return self.Kernel_Count

    @property
    def Input(self):
        return self.Previous_Channels

    def clear_parameters(self):
        self.Kernels = None
        self.Bias = None

    def calculate_logit(self, x):
        # Calculate convolution result for each sample each channel
        conv = np.asarray([[self.__channel_conv(xx, kernel) for kernel in self.Kernels] for xx in x])
        # Add bias
        for i in range(conv.shape[0]):
            for j in range(conv.shape[1]):
                conv[i,j] += self.Bias[j]
        # Swap channel axis to last dimension
        conv = np.swapaxes(np.swapaxes(conv, 1, 3), 1, 2)

        return conv

    def initialize_parameters(self, x):
        # init parameters if it hasn't been initialized
        if self.Kernels is None:
            # Record channel count of previous layer
            self.Previous_Channels = x.shape[-1]
            self.__init_kernel()
        if self.Bias is None:
            self.__init_bias()

        return None

    def delta_wb(self, x, gradient):
        # g = \partial \sigma (z) / \partial z
        act_grad = self.Act.gradient(self.logit(x))
        # apply
        gradient = np.multiply(act_grad, gradient)
        # for each weights
        # w[:,:,i] = g * rot180(x)[:,:,i]
        delta_w_single = lambda grad_map, input: [self.__channel_conv_revt(grad_map[:,:,j], input) for j in range(self.Kernel_Count)]
        # for all weights
        # w[j,:,:,:] = np.sum([np.swapaxes([g[q,:,:,j] * rot180(x[q,:,:,i]) for i in input_channels], 1, 3).swapaxes(1,2) for q in samples], axis=0)
        delta_w = np.sum([delta_w_single(gradient[i], x[i]) for i in range(gradient.shape[0])], axis=0)
        delta_b = gradient.sum(axis=1).sum(axis=1).sum(axis=0)

        return delta_w, delta_b, gradient

    def apply_wb(self, w, b, y):
        self.Kernels = self.Kernels - w
        self.Bias = self.Bias - b

        # sum all gradient calculated from single input channel
        grad_back_per_channel = lambda grad_map: np.sum([self.__channel_conv_upsample(grad_map, self.Kernels[:,:,:,i]) for i in range(self.Previous_Channels)], axis=0)
        # calculate for all samples
        grad_back = np.asarray([grad_back_per_channel(grad) for grad in y])

        return grad_back

    def F(self, x):
        """
            Calculate from input to output result
        """
        return self.Act.activation(self.logit(x))

    def backpropagation(self, x, gradient):
        """
            Back propagation algorithm
            gradient is 4-dimensional like: [samples, height, width, channel]
            x is 4-dimensional like: [samples, heights, width, channel]
            w is 3-dimensional like: [heights, width, channel]
            self.W is 4-dimensional like: [kernel, height, width, channel]
            b is 1-dimensional like: [feature_maps]
            self.B is 1-dimensional like: [kernel]
        """
        w, b, y = self.delta_wb(x, gradient)
        grad_back = self.apply_wb(w, b, y)

        return grad_back


class MaxPool(ILayer):

    def __init__(self, filter_size, strikes=None):
        """
            max pool layer
        :param filter_size: Shape of filter. [lines, columns] :list[int]
        :param strikes: Step per operation. [rows skip, columns skip] :list[int]
        """
        # pooling OP parameters
        super().__init__()
        if strikes is None:
            self.Strikes = filter_size
        else:
            self.Strikes = strikes
        # pooling Kernel parameters
        self.Filter_Size = filter_size

    def __check(self, x, filter_size=None, strikes=None):
        if filter_size is None:
            filter_size = self.Filter_Size
        if strikes is None:
            strikes = self.Strikes
        if (x.shape[0] % strikes[0] != 0) or (x.shape[1] % strikes[1] != 0):
            raise AssertionError('Cannot pooling input x shape like {} with filter shape like {}.'.format(x.shape, filter_size))

        return filter_size, strikes

    def __pool2d(self, x, filter_size=None, strikes=None, act=np.max):

        # check before do
        filter_size, strikes = self.__check(x, filter_size, strikes)

        # build buffer for result
        result_height = x.shape[0] // strikes[0]
        result_width = x.shape[1] // strikes[1]
        result = np.zeros(shape=[result_height, result_width])
        # calculate result
        for i in range(result_height):
            for j in range(result_width):
                # pooling window up
                row_start = i * strikes[0]
                # pooling window bottom
                row_end = row_start + filter_size[0]
                # window left
                col_start = j * strikes[1]
                # window right
                col_end = col_start + filter_size[1]
                # pooling with specified action
                result[i,j] = act(x[row_start:row_end, col_start:col_end])

        return result

    def __pool2d_revt(self, sample, x, filter_size=None, strikes=None, act=lambda x: np.unravel_index(np.argmax(x), x.shape)):

        # check before do
        filter_size, strikes = self.__check(sample, filter_size, strikes)

        # build result
        result = np.zeros_like(sample)
        # do up-sampling
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # pooling window up
                row_start = i * strikes[0]
                # pooling window bottom
                row_end = row_start + filter_size[0]
                # window left
                col_start = j * strikes[1]
                # window right
                col_end = col_start + filter_size[1]
                # get arg max
                idx = act(sample[row_start:row_end, col_start:col_end])
                # save
                result[row_start:row_end, col_start:col_end][idx] = x[i, j]

        return result

    @property
    def Trainable(self):
        return False

    @property
    def Input(self):
        return 'Doesn\'t matter'

    @property
    def Output(self):
        return 'Doesn\'t matter'

    def calculate_logit(self, x):
        # pooling each sample
        pool = np.asarray([[self.__pool2d(xx[:,:,channel]) for channel in range(xx.shape[-1])] for xx in x])
        pool = np.swapaxes(np.swapaxes(pool, 1, 3), 1, 2)

        return pool

    def F(self, x):
        # no activation function in pooling layer
        return self.logit(x)

    def delta_wb(self, x, gradient):
        return self.__pool2d_revt(x, gradient)

    def apply_wb(self, w, b, y):
        # no weights in pooling layer
        pass

    def backpropagation(self, x, gradient):
        # total result
        result = []
        # up-sampling gradient for each sample
        for i in range(x.shape[0]):
            channel_pooling = lambda channel:self.__pool2d_revt(x[i,:,:,channel], gradient[i,:,:,channel])
            # Add gradient
            result.append(np.swapaxes(np.swapaxes([channel_pooling(j) for j in range(x.shape[-1])], 0, 2), 0, 1))

        return np.asarray(result)


class Reshape(ILayer):

    def __init__(self, shape):
        """
            Reshape input while forward propagating
            Reshape back while backward propagating
        :param shape: shape of output
        """
        super().__init__()
        self.Shape_Out = shape
        self.Shape_In = None

    @property
    def Trainable(self):
        return False

    @property
    def Input(self):
        return self.Shape_In

    @property
    def Output(self):
        return self.Shape_Out

    def calculate_logit(self, x):
        if self.Shape_In is None:
            self.Shape_In = x.shape
            temp_out = [self.Shape_In[0]]
            temp_out.extend(self.Shape_Out)
            self.Shape_Out = temp_out
        return np.reshape(x, self.Shape_Out)

    def F(self, x):
        # no activation function in pooling layer
        return self.logit(x)

    def delta_wb(self, x, gradient):
        return np.reshape(gradient, self.Shape_In)

    def apply_wb(self, w, b, y):
        pass

    def backpropagation(self, x, gradient):
        return self.delta_wb(x, gradient)