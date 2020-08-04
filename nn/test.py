from dataset.cifar import load_data
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
from keras.layers import *

if __name__ == '__main__':

    model = keras.Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=[32, 32, 3]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # model.fit()

    # model = SequentialModel_v2(logger=Logger('Test'))
    # model.add(FCLayer_v2(784, act=Tanh()))
    # model.add(FCLayer_v2(784, act=Tanh()))
    # model.add(FCLayer_v2(392, act=Tanh()))
    # model.add(FCLayer_v2(196, act=Tanh()))
    # model.add(FCLayer_v2(128, act=Tanh()))
    # model.add(FCLayer_v2(10, act=SoftmaxNoGrad()))
    #
    # model.compile(optimizer=GradientDecentOptimizer_v2(learn_rate=0.05),
    #               loss=CrossEntropyLossWithSoftmax(),
    #               metrics=[CategoricalAccuracy()])
    #
    # model.summary()
    #
    x, y = load_data(kind='train')
    x = x.reshape([-1, 32, 32, 3])
    model.fit(x, y, batch_size=128, epochs=50)
    #
    x_test, y_test = load_data(kind='t10k')
    x_test = x_test.reshape([-1, 32, 32, 3])
    result = model.evaluate(x_test, y_test)
    #
    # print('Evaluate result: {}'.format(encode(zip(model.History_Title[-len(result):], result))))


# if __name__ == '__main__':
#
#     model = SequentialModel_v2(logger=Logger('Test'))
#     model.add(Conv2dLayer([5,5], 16, 'SAME', [1,1]))
#     model.add(MaxPool([2,2], [2,2]))
#     model.add(Conv2dLayer([5,5], 16, 'SAME', [1,1]))
#     model.add(MaxPool([2,2], [2,2]))
#     model.add(Reshape(shape=[7*7*16]))
#     model.add(FCLayer_v2(1024, act=Tanh()))
#     model.add(FCLayer_v2(10, act=SoftmaxNoGrad()))
#
#     model.compile(GradientDecentOptimizer_v2(), CrossEntropyLossWithSoftmax(),
#                   [CategoricalAccuracy()])
#
#     model.summary()
#
#     x, y = load_mnist(kind='train')
#     x = x.reshape([-1, 28, 28, 1])
#     model.fit(x, y, batch_size=128, epochs=10)