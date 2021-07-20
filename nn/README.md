

* [Parallel\-SGD\-docs\-zh\_CN](#parallel-sgd-docs-zh_cn)
  * [nn(神经网络)](#nn%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
    * [神经网络使用流程](#%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%BD%BF%E7%94%A8%E6%B5%81%E7%A8%8B)
      * [1\. build](#1-build)
        * [1\.1\.完全自定义网络](#11%E5%AE%8C%E5%85%A8%E8%87%AA%E5%AE%9A%E4%B9%89%E7%BD%91%E7%BB%9C)
        * [1\.2 SequentialModel](#12-sequentialmodel)
        * [1\.3 model\.Model](#13-modelmodel)
        * [1\.4 完全自定义网络](#14-%E5%AE%8C%E5%85%A8%E8%87%AA%E5%AE%9A%E4%B9%89%E7%BD%91%E7%BB%9C)
      * [2\. setup](#2-setup)
      * [3\. compile](#3-compile)
      * [4\. fit](#4-fit)
      * [5\. evaluate](#5-evaluate)
      * [6\. predict](#6-predict)
      * [7\. save&amp;load](#7-saveload)
    * [数据集处理](#%E6%95%B0%E6%8D%AE%E9%9B%86%E5%A4%84%E7%90%86)
  * [API](#api)
    * [1\. activation](#1-activation)
      * [1\.1 接口定义](#11-%E6%8E%A5%E5%8F%A3%E5%AE%9A%E4%B9%89)
      * [1\.2 实现类](#12-%E5%AE%9E%E7%8E%B0%E7%B1%BB)
    * [2\. layer](#2-layer)
      * [2\.1 接口定义](#21-%E6%8E%A5%E5%8F%A3%E5%AE%9A%E4%B9%89)
      * [2\.2 实现类](#22-%E5%AE%9E%E7%8E%B0%E7%B1%BB)
    * [3\. loss](#3-loss)
      * [3\.1 接口定义](#31-%E6%8E%A5%E5%8F%A3%E5%AE%9A%E4%B9%89)
      * [3\.2 实现类](#32-%E5%AE%9E%E7%8E%B0%E7%B1%BB)
    * [4\. metric](#4-metric)
      * [4\.1 接口定义](#41-%E6%8E%A5%E5%8F%A3%E5%AE%9A%E4%B9%89)
      * [4\.2 实现类](#42-%E5%AE%9E%E7%8E%B0%E7%B1%BB)
    * [5\. model](#5-model)
      * [5\.1 接口定义](#51-%E6%8E%A5%E5%8F%A3%E5%AE%9A%E4%B9%89)
    * [6\. gradient\_descent](#6-gradient_descent)
      * [6\.1 接口定义](#61-%E6%8E%A5%E5%8F%A3%E5%AE%9A%E4%B9%89)
      * [6\.2 实现类](#62-%E5%AE%9E%E7%8E%B0%E7%B1%BB)

# Parallel-SGD-docs-zh_CN

## nn(神经网络)
> 本模块是一个图神经网络模块，可供用户搭建自己的神经网络，提供了常见的数据集处理、激活函数、层、优化器、损失函数、衡量指标、自动求导功能。整体设计保留了**大量接口**可供用户自定义实现自己的想法创意。

### 神经网络使用流程

#### 1. build
##### 1.1.完全自定义网络
   最简单的使用建立网络的方式就是调用框架已实现的经典网络。
```python
model = nn.model.dnn.DNN(input_shape=(-1,784))
```

##### 1.2 SequentialModel

   该方式的使用类似`tf.keras.models.Sequential()`,该方法使用简单，但是只能表示线性的神经网络。下面展示如何使用该方法搭建网络
```python
import nn

model = nn.model.SequentialModel(input_shape=[-1, 32, 32, 3])
model.add(nn.layer.Conv2D(kernel=64, kernel_size=[3, 3], activation=nn.activation.LeakReLU()))
model.add(nn.layer.Conv2D(kernel=64, kernel_size=[3, 3], activation=nn.activation.LeakReLU()))
model.add(nn.layer.Conv2D(kernel=64, kernel_size=[3, 3], activation=nn.activation.LeakReLU()))
model.add(nn.layer.Conv2D(kernel=64, kernel_size=[3, 3], activation=nn.activation.LeakReLU()))
model.add(nn.layer.Flatten())
model.add(nn.layer.Dense(units=128, activation=nn.activation.HTanh()))
model.add(nn.layer.Dense(units=10, activation=nn.activation.Softmax()))
```
##### 1.3 model.Model
   该方式的使用类似`tf.keras.Model`,通过继承该类可以搭建更加复杂流程的神经网络，下面展示这种方式搭建网络
```python
class DNN(Model):

    def __init__(self, input_shape: [Tuple[int]] = None):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        fc1 = Dense(inputs=x, activation=Tanh(), units=784)
        self.__var_list.extend(fc1.variables)

        fc2 = Dense(inputs=fc1, activation=Tanh(), units=784)
        self.__var_list.extend(fc2.variables)

        fc3 = Dense(inputs=fc2, activation=Tanh(), units=392)
        self.__var_list.extend(fc3.variables)

        dropout = Dropout(inputs=fc3)

        fc4 = Dense(inputs=dropout, activation=Tanh(), units=128)
        self.__var_list.extend(fc4.variables)

        fc5 = Dense(inputs=fc4, activation=Softmax(), units=10)
        self.__var_list.extend(fc5.variables)

        return fc5
```
##### 1.4 完全自定义网络
   本模块支持自动求导，所以用户可以实现不依赖神经网络层任意自动求导，并且使用优化器对其进行梯度下降。下面展示如何使用自动求导功能。

```python
class LR(nn.model.Model):

    def __init__(self):
        super().__init__()
        self.w = nn.Variable(shape=[1,1])
        self.b = nn.Variable(shape=[1])

    def call(self, x):
        return x * self.w + self.b

    def trainable_variables(self):
        return self.w, self.b
```

#### 2. setup
   模型建立后首先需要绑定模型优化使用的损失函数，以及衡量指标（可选，变长），这里计入的衡量指标将在训练以及评估阶段显示。

```python
model.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())
```

#### 3. compile
   该步骤用于选择模型更新所用的优化器，其中优化器划分为两个层次，一个负责如何使用梯度，另一个用于计算梯度，如果使用的是梯度下降策略，使用adam计算梯度，选择额外的指定学习率,当然也可以简化写法，如下面所示。

```python
model.compile(Optimize(nn.optimizer.GDOptimizer, nn.gradient_descent.ADAMOptimizer, gd_params=(1e-3, )))
model.compile(nn.gradient_descent.ADAMOptimizer)
```
#### 4. fit
   该步骤非常类似keras的使用，这里封装了各种循环，放入样本、标签，epoch，batch_size即可训练，训练时会展示setup阶段选择的batch的评估指标。

```python
model.fit(x, label=y, epoch=1, batch_size=64)
```
#### 5. evaluate
   该步骤用于评估模型在测试集上的表现，输出指标依然由setup阶段确认

```python
model.evaluate(x_t, y_t)
```
#### 6. predict
   该步骤用于输出测试结果y。

```python
model.predict(x_t)
```

#### 7. save&load
   本步骤用于保存/恢复 模型结构、参数、setup和compile的参数，便于保存实验结果。

```python
model.save('abc.model')
model = nn.model.Model.load('abc.model')
```
### 数据集处理
   数据集的处理采用了链式的处理流，方便在与本项目其他模块进行协作，现阶段处理的操作较少，只包括了打乱数据集、图片的简单处理、**数据集非独立同分布化**，数据集的处理过程如下

```python
from nn.dataset.transforms import ImageCls, Shuffle
from nn.dataset import CIFAR

trans = Shuffle().add(ImageCls())
x, y, _, _ = trans(*CIFAR().load())
```



## API

### 1. activation
#### 1.1 接口定义
   激活函数接口继承了一元操作符：`AbsFlexibleUnaryNode`和激活操作:`IActivation`，下面为相关接口（不要慌），实际使用时，只需要写do_forward、do_backward就能重新写一个激活函数。
```python
class AbsActivation(AbsFlexibleUnaryNode, IActivation):

    def output_shape(self):
        return None

class IActivation(metaclass=ABCMeta):

    @abstractmethod
    def do_forward(self, x: [float, ndarray], training: bool = True) -> [float, ndarray]:
        """
            Do forward propagation.
        """
        pass

    @abstractmethod
    def do_backward(self, x: [float, ndarray], grad: [float, ndarray]) -> [ndarray, float]:
        """
            Do backward propagation.
        """
        pass

class AbsFlexibleUnaryNode(IUnaryNode, IFlexNode):

    def __init__(self, op: [IOperator] = None):
        self.__op_child: IOperator = op
        self.__ref_input: [ndarray, float] = 0

    @property
    def op_child(self):
        return self.__op_child

    def set_input(self, op: IOperator):
        self.__op_child = op

    @abstractmethod
    def do_forward(self, x: [float, ndarray], training: bool = True) -> [float, ndarray]:
        """
            Do forward propagation.
        """
        pass

    @abstractmethod
    def do_backward(self, x: [float, ndarray], grad: [float, ndarray]) -> [ndarray, float]:
        """
            Do backward propagation.
        """
        pass

    def F(self, x: [float, ndarray, tuple] = None, state: ModelState = ModelState.Training) -> [float, ndarray]:
        """
            Forward propagate to get predictions.
        :return: output_ref
        """
        if self.__op_child:
            self.__ref_input = self.op_child.F(x, state)
        else:
            self.__ref_input = x
        self.do_forward(self.__ref_input, state == ModelState.Training)

    def G(self, grad: [float, ndarray] = None) -> None:
        """
            Backward propagate and update variables.
        :param grad: gradients of backward_predict layers
        """
        grad_back = self.do_backward(self.__ref_input, grad)
        if self.__op_child:
            self.op_child.G(grad_back)

    def clear_unused(self):
        pass

    def __getstate__(self):
        self.__ref_input = 0
        self.clear_unused()
        return self.__dict__
```

   如果想要定义新的激活函数只需要实现\_\_init\_\_、output_shape（基本是固定写法，除非激活函数还能改变尺寸）、do_forward、do_backward、clear_unused（基本为固定写法，用于保存模型时丢弃激活函数参数，减少存储（存储空间大的可以不丢））。下面以最常见的relu为例。

```python
class ReLU(AbsActivation):

    def __init__(self, op: IOperator = None):
        super().__init__(op)
        # 用于记录被丢弃的位置
        self.__ref_input: [np.ndarray] = None
    # 这个是从`AbsFlexibleUnaryNode`继承来的，可以拿到操作数的尺寸。
    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x, training=True):
        self.__ref_input = x.copy()
        self.__ref_input[self.__ref_input < 0] = 0
        return self.__ref_input

    def do_backward(self, x, grad):
        return np.multiply(grad, self.__ref_input >= 0)

    def clear_unused(self):
        self.__ref_input = None
```

#### 1.2 实现类
   现阶段以实现下列激活函数，调用方式如下。

```python
from nn.activation import ReLU, Sigmoid, Tanh, LeakReLU, Softmax, Linear, HTanh, SigmoidNoGrad
```

### 2. layer
#### 2.1 接口定义
​    layer接口`AbsLayer`继承了操作符：`IOperator`和懒加载初始化接口:`ILazyInitialization`，下面为`AbsLayer`接口。
```python
class AbsLayer(IOperator, ILazyInitialization):
    """
        Used for lazy initialization.
    """

    def __init__(self, inputs: IOperator = None, activation: IActivation = None):
        """
            Abstract layer class
        :param inputs: input operator, IOperator instance
        """
        self.__op_input = inputs
        self.__ref_input = None
        self.__activation = activation if activation else Linear()
        self.__initialized = False

    @property
    def input_ref(self):
        return self.__ref_input

    def set_input(self, inputs: IOperator):
        self.__op_input = inputs

    def __getstate__(self):
        self.__ref_input = None
        return self.__dict__

    @property
    @abstractmethod
    def variables(self) -> Iterable[ITrainable]:
        """
            Trainable units within this scope.
        :return: tuple
        """
        pass

    @abstractmethod
    def initialize_parameters(self, x) -> None:
        """
            Initialize parameters with given input_ref (x)
        :param x: ndarray
        """
        pass

    @abstractmethod
    def do_forward_predict(self, x):
        """
            Do forward propagate with given input_ref.
        :param x: ndarray
        """
        pass

    @abstractmethod
    def do_forward_train(self, x):
        """
            Do forward propagate with given input_ref.
        :param x: ndarray
        """
        pass

    @abstractmethod
    def backward_adjust(self, grad) -> None:
        """
            Backward propagate with weights adjusting.
        :param grad: ndarray
        """
        pass

    @abstractmethod
    def backward_propagate(self, grad):
        """
            Backward propagate.
        :param grad: ndarray
        :return: return the gradient from backward to input_ref (x)
        """
        pass

    def reset(self):
        self.__initialized = False

    def __forward_prepare(self, x):
        self.initialize_parameters(x)
        self.__initialized = True

    def F(self, x: [float, ndarray, tuple] = None, state: ModelState = ModelState.Training) -> Union[float, ndarray]:
        """
            Do forward propagate.
        :param x: input of this layer.
                This parameter works only when this layer is not part of the computation graph.
        :param state: State to identify training process, works in some particular layer like
                (Dropout).
        :return: output of this layer.
        """
        self.__ref_input = self.__op_input.F(x, state) if self.__op_input else x
        if not self.__initialized:
            self.__forward_prepare(self.__ref_input)
        if state != ModelState.Training:
            return self.__activation.do_forward(self.do_forward_predict(self.__ref_input))
        else:
            return self.__activation.do_forward(self.do_forward_train(self.__ref_input))

    def G(self, grad: [float, ndarray] = None) -> None:
        """
            Do backward and adjust parameters.
        :param grad: Gradients from back-propagation, set to None when this layer doesnt needs
                input gradients. e.g. loss functions.
        :return: None, try get gradients from placeholder or variable.
        """
        # adjust variables with given gradients.
        gradient = self.__activation.do_backward(None, grad)
        # adjust previous layers.
        if self.__op_input:
            self.__op_input.G(self.backward_propagate(gradient))
        # adjust current layer.
        self.backward_adjust(gradient)
```

   如果想要定义新的层需要实现\_\_init\_\_、output_shape、variables、do_backward、initialize_parameters、do_forward_predict、backward_adjust、backward_propagate、\_\_str\_\_、\_\_repr\_\_。下面以常见的dense为例。

```python
class Dense(AbsLayer):

    def __init__(self, units, activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__layer_units = units
        self.__w = Weights()
        self.__b = Weights()

    def output_shape(self) -> [list, tuple, None]:
        return [-1, self.__layer_units]

    @property
    def variables(self) -> tuple:
        return self.__w, self.__b

    def initialize_parameters(self, x) -> None:
        high = np.sqrt(6 / (x.shape[1] + self.__layer_units))
        low = -high
        self.__w.set_value(np.random.uniform(low=low, high=high, size=[x.shape[1], self.__layer_units]))
        self.__b.set_value(np.zeros(shape=[self.__layer_units]))
	
    # 进行前向传播，预测与训练在某些层（比如dropout）上表现不同
    def do_forward_predict(self, x):
        return np.dot(x, self.__w.get_value()) + self.__b.get_value()

    def do_forward_train(self, x):
        return self.do_forward_predict(x)
        
	# 根据后面层传来的梯度更新自己的权值
    def backward_adjust(self, grad) -> None:
        g_w = np.dot(self.input_ref.T, grad)
        self.__w.adjust(g_w)
        self.__b.adjust(grad)

	# 继续进行反向传播
    def backward_propagate(self, grad):
        g_x = np.dot(grad, self.__w.get_value().T)
        return g_x

    def __str__(self):
        return "<Dense Layer, Units: {}>".format(self.__layer_units)

    def __repr__(self):
        print(self.__str__())
```

#### 2.2 实现类
   现阶段以实现下列layer，调用方式如下。

```python
from nn.layer import Conv2D, MaxPool, Reshape, Dense, Flatten, Dropout, BatchNorm
```
### 3. loss
#### 3.1 接口定义
​    loss接口`ILoss`实现了`IMetric`。
```python
class ILoss(IMetric):

    @abstractmethod
    def gradient(self, left:[float, ndarray], right:[float, ndarray]) -> (ndarray, ndarray):
        """
            Calculated gradient of L(x, y) for both x and y.
        :param left: left input
        :param right: right input
        :return: tuple for left gradient and right gradient
        """
        pass

    def description(self):
        return "Loss"

class IMetric(metaclass=ABCMeta):

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
```

   如果想要定义新的损失函数需要实现\_\_init\_\_、gradient、metric、_\_str\_\_、\_\_repr\_\_。下面以常见的MSELoss为例。

```python
class MSELoss(ILoss):

    def __init__(self):
        pass

    def __repr__(self):
        print(self.__str__())

    def __str__(self):
        return "<Mean Square Error Loss>"
	
    def gradient(self, arg1, arg2):
        return 2.0 * (arg1 - arg2), -2.0 * (arg1 - arg2)
	
    # 相当于前向传播
    def metric(self, arg1, arg2):
        return np.mean(np.square(arg1 - arg2))
```

#### 3.2 实现类
   现阶段以实现下列loss，调用方式如下。

```python
from nn.loss import MSELoss, Cross_Entropy_With_Softmax, Cross_Entropy, TanhLoss
```


### 4. metric
#### 4.1 接口定义
​    metric接口`IMetric`没有更上层的继承，里面也比较简单，一个自描述的方法，还有一个衡量指标功能。
```python
class IMetric(metaclass=ABCMeta):

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
```

   如果想要定义新的衡量指标需要实现\_\_init\_\_、metric、description（实际就实现一个metric就行）。下面以常见的`CategoricalAccuracy`为例。

```python
class CategoricalAccuracy(IMetric):
    """
        Categorical Accuracy metric.
        Use one-hot vector as label.

        Metric can be used in MLR, CNN.
    """

    def __init__(self):
        pass

    def metric(self, y, label):
        y = (y == y.max(axis=1).reshape([-1, 1])).astype('int')
        label = label.astype('int')
        result = np.sum(y & label) / len(y)
        return result

    def description(self):
        return 'accuracy'
```

#### 4.2 实现类
   现阶段以实现下列metric，调用方式如下。

```python
from nn.metric import MeanSquareError, CategoricalAccuracy, BinaryAccuracy, RelativeError, MeanSquareError, RelativeMeanSquareError, EqualErrorRate, TruePositive, FalsePositive, TrueNegative, TrueNegative, FalseNegative, AreaUnderCurve
```
### 5. model
#### 5.1 接口定义
​    model接口`IModel`定义如下，可以从该接口中看到模型的各种方法。
```python
class IModel(metaclass=ABCMeta):

    @abstractmethod
    def setup(self, loss: ILoss, *metrics: IMetric):
        """
             loss and metrics
        :param loss: ILoss
        :param metrics: IMetric
        :return: None
        """
        pass

    @abstractmethod
    def compile(self, optimizer: Union[IOpContainer, Type[IGradientDescent]]):
        """
            Compile model with given optimizer
        :param optimizer: IOptimizer
        :return: None
        """
        pass

    @abstractmethod
    def fit(self, x: [ndarray, IDataFeeder], label: [ndarray] = None, epoch: int = 4, batch_size: int = 64,
            printer: IPrinter = None) -> FitResultHelper:
        """
            Fit model with given samples.
        :param x: ndarray or data feeder. requires a IDataFeeder instance or both x and label for ndarray instance.
        :param epoch: int, Epoch of training
        :param label: ndarray, Label of samples
        :param batch_size: int, batch size
        :param printer: printer type
        :return: Fitting result, contains all history records.
        """
        pass

    @abstractmethod
    def fit_history(self) -> FitResultHelper:
        """
            Get all history records.
        :return:
        """
        pass

    @abstractmethod
    def evaluate(self, x: ndarray, label: ndarray) -> Dict[str, float]:
        """
            Evaluate this model with given metric.
        :param x: input samples
        :param label: labels
        :return: evaluation result
        """
        pass

    @abstractmethod
    def predict(self, x: ndarray) -> ndarray:
        """
            Predict give input
        :param x: input samples
        :return:
        """
        pass

    @abstractmethod
    def clear(self):
        """
            Clear and reset model parameters.
        :return:
        """
        pass

    @abstractmethod
    def summary(self) -> str:
        """
            Return the summary string for this model.
        :return: String
        """
        pass

class Model(IModel):

    def __init__(self, input_shape: [Tuple[int]] = None):
        self.__placeholder_input = Placeholder(input_shape)
        self.__ref_output: [IOperator] = None
        self.__metrics: List[IMetric] = []
        self.__loss: [ILoss] = None
        self.__optimizer: [IOptimizer] = None
        self.__fit_history: FitResultHelper = FitResultHelper()

    @abstractmethod
    def trainable_variables(self) -> Iterable[ITrainable]:
        pass

    @property
    def is_setup(self):
        return isinstance(self.__loss, ILoss) and isinstance(self.__ref_output, IOperator)

    @property
    def can_fit(self):
        return self.is_setup and isinstance(self.__optimizer, IOpContainer)

    @abstractmethod
    def call(self, x: Placeholder) -> IOperator:
        pass

    @property
    def loss(self):
        return self.__loss

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def metrics(self):
        return self.__metrics

    def setup(self, loss: ILoss, *metrics: IMetric):
        if self.__ref_output is None:
            self.__ref_output = self.call(self.__placeholder_input)
        # validate model
        if self.__placeholder_input.get_shape() is not None:
            self.__placeholder_input.set_value()
            # reset and validate
            self.__ref_output.F()
        # setup loss
        self.__loss: ILoss = loss
        # setup metric
        self.__metrics = [self.__loss]
        self.__metrics.extend(metrics)
        # validate metrics and set title
        title = ["Epochs", "Batches", "in", "Total"]
        for metric in self.__metrics:
            assert isinstance(metric, IMetric), "Something cannot be interpreted as metric were passed in."
            title.append(metric.description())

        # set title
        self.__fit_history.set_fit_title(title)

    def compile(self, optimizer: Union[IOpContainer, Type[IGradientDescent]]):
        # set optimizer
        if isinstance(optimizer, IOpContainer):
            self.__optimizer = optimizer
        else:
            self.__optimizer = OpContainer(GDOptimizer, optimizer)
        self.__optimizer.optimize(*self.trainable_variables())

    def __evaluate_metrics(self, y, label) -> list:
        return [metric.metric(y, label) for metric in self.__metrics]

    def fit(self, x: [ndarray, IDataFeeder], label: [ndarray] = None, epoch: int = 4, batch_size: int = 64,
            printer: IPrinter = None) -> FitResultHelper:
        assert self.can_fit, "Model is not prepared for training."
        assert isinstance(x, IDataFeeder) or label is not None, "Fitting process requires both x and label."

        if isinstance(x, ndarray):
            x = NumpyDataFeeder(x, label, batch_size=batch_size)

        self.__optimizer.set_batch_size(x.batch_size)
        title = [metric.description() for metric in self.__metrics]

        for j in range(epoch):
            epoch_rec = np.zeros(shape=[len(title)])
            for part_x, part_y in x:
                self.__placeholder_input.set_value(part_x)
                # do forward propagation
                y = self.__ref_output.F()
                # get loss
                grad_y, _ = self.__loss.gradient(y, part_y)
                # do backward propagation from loss
                self.__ref_output.G(grad_y)
                # record fitting process
                batch_rec = self.__evaluate_metrics(y, part_y)
                fit_rec = [j + 1, x.position, x.length, self.__fit_history.count + 1]
                fit_rec.extend(batch_rec)
                epoch_rec += np.asarray(batch_rec) / x.length

                str_formatted = self.__fit_history.append_row(fit_rec)
                if printer:
                    printer.log_message(str_formatted)
                else:
                    # get stdout
                    sys.stdout.write('\r' + str_formatted)
                    sys.stdout.flush()
            print('')
            str_formatted = ["\t{}:{:.2f}".format(name, val) for name, val in zip(title, epoch_rec)]
            print("Epoch Summary:{}".format(','.join(str_formatted)))

        return self.__fit_history

    def fit_history(self) -> FitResultHelper:
        return self.__fit_history

    def evaluate(self, x: ndarray, label: ndarray) -> Dict[str, float]:
        assert self.is_setup, "Model hasn't setup."
        x = NumpyDataFeeder(x, label, batch_size=100)
        # get stdout
        import sys
        # get title
        title = [metric.description() for metric in self.__metrics]
        eval_recs = []
        for part_x, part_y in x:
            # set placeholder
            self.__placeholder_input.set_value(part_x)
            # do forward propagation
            y = self.__ref_output.F(state=ModelState.Evaluating)
            # get evaluation
            eval_rec = self.__evaluate_metrics(y, part_y)
            eval_recs.append(eval_rec)
            str_formatted = ["\t{}:{:.2f}".format(name, val) for name, val in zip(title, np.mean(eval_recs, axis=0))]
            sys.stdout.write("\rEvaluating: {:.2f}%{}.".format(100 * x.position / x.length, ','.join(str_formatted)))
            sys.stdout.flush()
        # flush a new line
        print('')
        return dict(zip(title, np.mean(eval_recs, axis=0)))

    def predict(self, x: ndarray):
        self.__placeholder_input.set_value(x)
        y = self.call(self.__placeholder_input).F(state=ModelState.Predicting)
        return y

    def clear(self):
        for var in self.trainable_variables():
            var.reset()

    def summary(self) -> str:

        summary = '\n------------\t\tModel Summary\t\t------------\n'

        summary += "No structure description available for this model.\n"

        if self.__loss and self.__optimizer and self.__metrics:
            summary += '\t------------\t\tAppendix\t\t------------\n'
            summary += '\tLoss:\n\t\t{}\n'.format(self.__loss)
            summary += '\tOptimizer:\n\t\t{}\n'.format(self.__optimizer)
            summary += '\tMetrics:\n'
            for metric in self.__metrics:
                summary += '\t\t<Metric: {}>\n'.format(metric.description())
            summary += '\t------------\t\tAppendix\t\t------------\n'
        summary += '\n------------\t\tModel Summary\t\t------------\n'

        return summary

    def save(self, file: str):
        with open(file, 'wb') as fd:
            pickle.dump(self, fd)

    @staticmethod
    def load(file: str) -> 'Model':
        with open(file, 'rb') as fd:
            model = pickle.load(fd)
        if model.__optimizer:
            model.compile(model.__optimizer)
        return model
```

   如果想要定义新的模型可以参考上面的神经网络使用流程。

### 6. gradient_descent

#### 6.1 接口定义
​     gradient_descent接口`IGradientDescent`用于接受梯度，然后如何加工这个梯度，之前提到过本项目将优化器抽象成两个层次，本部分负责加工梯度，在其之上的是optimizer用来使用梯度，决定是梯度下降还是梯度上升（别问为什么有梯度上升这么傻吊的想法）。所以如果想要重写优化器，只要重写本子模块就可以了。
```python
class IGradientDescent(metaclass=ABCMeta):

    @abstractmethod
    def delta(self, var):
        """
            Calculate only incremental value, do not update.
        :param var: variable or placeholder
        :return: delta w
        """
        pass
```

   如果想要定义新的优化器（额，我感觉可以叫优化器吧。。。）需要实现\_\_init\_\_、delta、\_\_str\_\_。下面以常见的`SGDOptimizer`为例。

```python
class SGDOptimizer(IGradientDescent):

    def __init__(self, learn_rate=0.01):
        self.__learn_rate = learn_rate

    # SGD只要乘个学习率就返回去就行了，如果想整带有记录的就可以在init开变量存一存
    def delta(self, gradient):
        return gradient * self.__learn_rate

    def __str__(self):
        return "<SGD Optimizer>"
```

#### 6.2 实现类
   现阶段以实现下列gradient_descent，调用方式如下。

```python
from nn.gradient_descent import ADAMOptimizer, SGDOptimizer, AdaGradOptimizer, AdaDeltaOptimizer, RMSPropOptimizer
```

