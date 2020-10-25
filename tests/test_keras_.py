from tensorflow import keras
from dataset import CIFAR, MNIST
from dataset.transforms import ImageCls

if __name__ == '__main__':
    x, y, x_t, y_t = ImageCls()(*MNIST().load())

    model = keras.Sequential()
    model.add(keras.layers.Reshape(target_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=[5, 5], activation='relu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=[5, 5], activation='relu'))
    # model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(filters=64, kernel_size=[5, 5], activation='relu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=[5, 5], activation='relu'))
    # model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128, activation='tanh'))
    model.add(keras.layers.Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(x, y, epochs=40, batch_size=100)
