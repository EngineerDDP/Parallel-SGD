from tensorflow import keras
from dataset import CIFAR
from dataset.transforms import ImageCls

if __name__ == '__main__':
    x, y, x_t, y_t = ImageCls()(*CIFAR().load())
    model = keras.Sequential()
    model.add(keras.layers.Reshape(target_shape=(32, 32, 3)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=[3, 3], activation='relu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=[3, 3], activation='relu'))
    # model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(filters=64, kernel_size=[3, 3], activation='relu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=[3, 3], activation='relu'))
    # model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128, activation='tanh'))
    model.add(keras.layers.Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x, y, epochs=5, batch_size=64)
