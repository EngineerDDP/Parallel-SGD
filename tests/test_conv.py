import nn
from dataset import CIFAR, MNIST
from dataset.transforms import ImageCls, Shuffle
from nn.layer import Conv2D, Reshape, Dense, Flatten
from nn.loss import Cross_Entropy_With_Softmax
from nn.metric import CategoricalAccuracy, np
from nn.model import SequentialModel
import tensorflow as tf

if __name__ == '__main__':
    # r_x = np.random.uniform(low=-1, high=1, size=[10, 5, 5, 10])
    # r_w = np.random.uniform(low=-1, high=1, size=[3, 3, 10, 10])
    # # r_x = np.arange(0, 2.5, 0.1).reshape([1,5,5,1])
    # # r_w = np.arange(0, 0.9, 0.1).reshape([3,3,1,1])
    #
    # x = tf.Variable(tf.constant(r_x))
    # w = tf.Variable(tf.constant(r_w))
    # # y = tf.ones(shape=[])
    # with tf.GradientTape() as tape:
    #     y_ = tf.nn.conv2d(x, w, strides=1, padding="VALID")
    #     label = tf.reduce_sum(y_)
    #
    # y_g = tf.ones_like(y_)
    # grad = tape.gradient(label, w)
    #
    # tf_out = tf.nn.conv2d(tf.transpose(x[:, :, :, :], perm=[3, 1, 2, 0]),
    #                       tf.transpose(y_g, perm=[1, 2, 0, 3]), 1, "VALID")
    # out = tf.transpose(tf_out, perm=[1, 2, 0, 3])
    #
    # err = np.sum(np.abs(grad.numpy() - out.numpy()))
    # print(grad[:, :, 0, 0])
    # print(out[:, :, 0, 0])

    # model = SequentialModel()
    # # model.add(Reshape(shape=[-1, 28, 28, 1]))
    # model.add(Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
    # model.add(Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
    # model.add(Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
    # model.add(Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
    # model.add(Flatten())
    # model.add(Dense(units=128, activation=nn.activation.HTanh(leak_coefficient=0.2)))
    # model.add(Dense(units=10, activation=nn.activation.Softmax()))

    model = nn.model.Model.load("CIFARNET.model")

    trans = ImageCls()

    x, y, x_t, y_t = trans(*CIFAR().load())
    # x = x[:10000]
    # y = y[:10000]
    model.setup(Cross_Entropy_With_Softmax(), CategoricalAccuracy())
    model.compile(nn.gradient_descent.ADAMOptimizer())
    model.fit(x, label=y, epoch=10, batch_size=100)
    model.evaluate(x, y)

    # import tensorflow as tf
    #
    # x = tf.constant(np.arange(0, 2.5, 0.1).reshape([1,5,5,1]))
    # y = tf.constant(np.arange(0, 0.9, 0.1).reshape([1,3,3,1]))
    # w = tf.constant(np.ones(shape=[3,3,1,1]))
    #
    # x_ = tf.nn.conv2d_transpose(y, w, x.shape, [1,1], padding="VALID")
    #
    # print(x_.shape)
