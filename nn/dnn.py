import numpy as np

raise NotImplementedError()

class RobustLinearRegressionCPU:

    def __init__(self, epoches = 100, decentlimit=1e-5):
        '''
            epoches for maxmum epoches in training steps
            decentlimit for minimum step of loss reduction 

            training process will be aborted if any of above limits no longer satisfied
        '''
        self.LR = 0.1
        self.W = 0.2
        self.B = 0.2
        self.Epoches = epoches
        self.DecentLimit = decentlimit

    def __model(self, x):
        return self.W * x + self.B

    def fit(self, x, y):
        '''
            fit x --> y reflection with given samples
        '''
        Pre_loss = 0.0
        for i in range(self.Epoches):
            for xx,yy in zip(x,y):
                Loss = self.__train(xx, yy)
            if np.abs(Pre_loss - Loss) < self.DecentLimit:
                break
            Pre_loss = Loss

    def predict(self, x):
        '''
            predict with fited model reflection
        '''
        return self.__model(x)

    def __train(self, x, y):
        x = np.mean(x)
        y = np.mean(y)
        L = np.square(np.tanh(y - self.__model(x)))
        self.W = self.W - self.LR * (-2.0 * x * np.tanh(y - self.__model(x)) * (1 - L))
        self.B = self.B - self.LR * (-2.0 * np.tanh(y - self.__model(x)) * (1 - L))
        return L


class Linear:
    """
    Linear activation
    """

    def __init__(self):
        pass

    def activation(self, x):
        return x

    def gradient(self, x):
        return 1


class ReLU:
    """
    ReLU activation
    """
    def __init__(self):
        pass

    def activation(self, x):
        r = x.copy()
        r[r < 0] = 0
        return r

    def gradient(self, x):
        r = x.copy()
        r[r < 0] = 0
        r[r > 0] = 1
        return r


class Sigmoid:
    """
    Sigmoid type activation
    """

    def __init__(self, delta=0.0):
        self.Delta = delta

    def activation(self, x):
        return 1 / (1 + np.exp(-1 * (x + self.Delta)))

    def gradient(self, x):
        return np.multiply(self.activation(x), (1 - self.activation(x)))


class Tanh:
    """
    Hyperbolic tangent function
    """

    def __init__(self, **kwargs):
        pass

    def activation(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return 1 - np.multiply(self.activation(x), self.activation(x))


class FCLayer:

    def __init__(self, units, w_init=None, b_init=None, act=Linear()):

        # use lazy initialization
        if w_init is not None:
            self.W = w_init
        else:
            self.W = None

        if b_init is not None:
            self.B = b_init
        else:
            self.B = None

        self.Act = act
        self.Output = units

    def reset(self):
        self.W = None
        self.B = None

    def logit(self, x):
        """
            Calculate logit
        """
        # lazy initialization
        # w shape is [output, input]
        # b shape is [output]
        if self.W is None:
            high = np.sqrt(1 / x.shape[0])
            low = -high
            self.W = np.random.uniform(low=low, high=high, size=[self.Output, x.shape[0]])
        if self.B is None:
            self.B = np.zeros(shape=[self.Output, 1])

        return np.dot(self.W, x) + self.B

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
        y = np.multiply(act_grad, gradient)

        # adjust weight
        delta_weight = y.dot(x.T)
        delta_bias = y.sum(axis=1)

        return delta_weight, delta_bias, y


    def apply_wb(self, w, b, y):

        self.W = self.W - w
        self.B = self.B - b

        # calculate gradient for BP
        grad = self.W.transpose().dot(y)

        return grad


    def backpropagation(self, x, gradient):
        """
            Calculate gradient, adjust weight and bias and return gradients of this layer
            x shape=[input, samples count]
            grad shape=[output, samples count]
        """
        # calculate gradient
        act_grad = self.Act.gradient(self.logit(x))
        # y shape=[output, samples count]
        y = np.multiply(act_grad, gradient)

        # adjust weight
        batch_weight = y.dot(x.T) / y.shape[1]
        self.W = self.W - batch_weight
        # adjust bias
        self.B = self.B - y.mean(axis=1)
        # recalculate gradient to propagate
        grad = self.W.transpose().dot(y)
        return grad


class GradientDecentOptimizer:

    def __init__(self, loss, layers, learnrate=0.01):
        self.LR = learnrate
        self.Loss = loss
        self.Layers = layers

    def train(self, x, label):
        """
            train the network with labeled samples
        """

        # reshape x to [-1,1]

        x = np.asmatrix(x).T
        label = np.asmatrix(label).T

        # forward propagation

        intermediate = [x]
        for nn in self.Layers:
            intermediate.append(nn.F(intermediate[-1]))

        loss = self.Loss.loss(intermediate[-1], label)

        # apply learning rate

        self.Grad = self.LR * self.Loss.gradient(intermediate[-1], label)
        grad = self.Grad

        # backward propagation

        self.Layers.reverse()
        i = 2
        for nn in self.Layers:
            grad = nn.backpropagation(intermediate[-1 * i], grad)
            i += 1

        self.Layers.reverse()

        # return loss

        return np.mean(loss)

from time import sleep


class AsyncGradientDecentOptimizer:

    def __init__(self, loss, layers, tag, com, learnrate=0.01):
        """
            Distributed machine learning implementation
            tag: used to identify myself in cluster
            com: used to update and require parameters from cluster
        """
        self.LR = learnrate
        self.Loss = loss
        self.Layers = layers

        self.Tag = tag
        self.Com = com

    def train(self, x, label):
        """
            train the network with labeled samples
        """

        total = len(x)

        #if self.Tag.Node_No == 3:
        #    sleep(10)

        # get blocks
        x = x[self.Tag.getSliceWithinBatch()]
        label = label[self.Tag.getSliceWithinBatch()]

        x = np.asmatrix(x).T
        label = np.asmatrix(label).T

        # forward propagation

        intermediate = [x]
        for nn in self.Layers:
            intermediate.append(nn.F(intermediate[-1]))

        loss = self.Loss.loss(intermediate[-1], label)

        # apply learning rate

        self.Grad = self.LR * self.Loss.gradient(intermediate[-1], label)
        grad = self.Grad

        # backward propagation

        self.Layers.reverse()
        i = 2
        for nn in self.Layers:
            w, b, y = nn.delta_wb(intermediate[-1 * i], grad)
            self.Com.putWeights(w, self.Tag, 'w')
            self.Com.putWeights(b, self.Tag, 'b')

            w_new = self.Com.acquireweights(self.Tag, 'w') / total
            b_new = self.Com.acquireweights(self.Tag, 'b') / total

            grad = nn.apply_wb(w_new, b_new, y)

            self.Tag.incLayer()
            i += 1

        self.Layers.reverse()

        # return loss
        self.Tag.incBatch()

        return np.mean(loss)


class AdagradOptimizer(GradientDecentOptimizer):

    def __init__(self, loss, layers, learnrate=0.01):
        super().__init__(loss, layers, learnrate)
        self.Gt = 0
        self.delta = 1e-8

    def train(self, x, label):
        # update learning rate
        learn_rate = self.LR
        if self.Gt != 0:
            self.LR = self.LR / np.sqrt(self.Gt + self.delta)

        # train
        loss = super().train(x, label)
        # print(self.LR)

        # update Gt
        self.Gt = self.Gt + np.mean(np.square(self.Grad))
        self.LR = learn_rate

        return loss


#----------------------------------------------------------
#Build your own loss functions below

class ILoss:

    def __init__(self):
        pass

    def loss(self, y, label):
        raise NotImplementedError()

    def gradient(self, y, label):
        raise NotImplementedError()


class MseLoss:

    def __init__(self):
        pass

    def loss(self, y, label):
        return np.mean(np.square(label - y), axis=0)

    def gradient(self, y, label):
        return (label - y) * -2


class CrossEntropyLoss:

    def __init__(self):
        pass

    def loss(self, y, label):
        # multiply element-wise
        return np.mean(np.multiply(label, np.log(y)) + np.multiply((1 - label), np.log(1 - y))) * -1

    def gradient(self, y, label):
        return ((1 - label) / (1 - y) - label / y) / label.shape[1]


class CrossEntropyLossWithSigmoid:

    def __init__(self):
        pass

    def loss(self, y, label):
        # multiply element-wise
        return np.mean(np.multiply(label, np.log(y)) + np.multiply((1 - label), np.log(1 - y))) * -1

    def gradient(self, y, label):
        return np.multiply(y - label, 1 / np.multiply(y, 1 - y))


class TanhLoss:

    def __init__(self):
        pass

    def loss(self, y, label):
        return np.mean(np.square(np.tanh(label - y)))

    def gradient(self, y, label):
        return -2.0 * np.multiply(np.tanh(label - y), (1 - np.square(np.tanh(label - y))))


class L1Loss:

    def __init__(self):
        self.L2Loss = MseLoss()

    def loss(self, y, label):
        return np.mean(label - y)

    def gradient(self, y, label):

        grad = label - y
        grad[np.where(grad < 0)] = -1
        grad[np.where(grad > 0)] = 1
        grad = grad * -1 + self.L2Loss.gradient(y, label)

        return grad


#----------------------------------------------------------


class Model:

    def __init__(self, nn, optimizer, onehot=True, debug=True):
        self.NN = nn
        self.Optimizer = optimizer
        self.Onehot = onehot
        self.Debug = debug

    def fit(self, x, y, epochs, batch_size, miniloss=None, minideltaloss=None):
        import time

        if batch_size > len(x):
            batch_size = len(x)
        preloss = 0.0
        time_t = time.time()
        interval = 0
        batches = int(len(x) / batch_size)

        # train
        for j in range(epochs):
            for i in range(batches):
                start = i * batch_size % (len(x) - batch_size + 1)
                end = start + batch_size
                part_x = x[start:end]
                part_y = y[start:end]

                loss = self.Optimizer.train(part_x, part_y)

                interval = time.time() - time_t

                if self.Debug and interval > 5:
                    print('epochs: {}/{}, batches: {}/{}, loss: {:.4f}'.format(j+1, epochs, i+1, batches, loss))
                    self.evalute(part_x, part_y)
                    time_t = time.time()
            
            #if minideltaloss is not None and np.abs(loss - preloss) < minideltaloss:
            #    break
            #if miniloss is not None and np.abs(loss) < miniloss:
            #    break

            preloss = loss

        return loss

    def reset(self):
        
        for nn in self.NN:
            nn.reset()

    def copy(self):

        nns = []

        for nn in self.NN:
            nns.append(FCLayer(nn.Output, nn.W, nn.B, nn.Act))

        return Model(nns, self.Optimizer, self.Onehot, self.Debug)


    def predict(self, x):

        # transpose x
        x = np.asmatrix(x).T

        for layer in self.NN:
            x = layer.F(x)

        x = x.T.getA()

        return x

    def evalute(self, x, y):

        predict = self.predict(x)
        if self.Onehot:
            y = y.argmax(axis=1)
            predict = predict.argmax(axis=1)
        else:
            predict = np.round(predict)

        acc = np.mean(np.equal(y, predict))

        if self.Debug:
            print('Accuracy: {:.4f}'.format(acc))

        return acc





