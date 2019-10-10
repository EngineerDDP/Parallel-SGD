import numpy as np
import pandas as pd

import time


class Model:

    def __init__(self, nn, optimizer, onehot=True, debug=True):
        self.NN = nn
        self.Optimizer = optimizer
        self.Onehot = onehot
        self.Debug = debug

    def fit(self, x, y, epochs, batch_size, miniloss=None, minideltaloss=None):

        times_count = 0
        track_content_time = []
        track_content_epoch = []

        if batch_size > len(x):
            batch_size = len(x)
        preloss = 0.0
        time_t = time.time() - 6
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
                    print('epochs: {}/{}, batches: {}/{}, loss: {:.4f}'.format(j + 1, epochs, i + 1, batches, loss))
                    self.evalute(part_x, part_y)
                    track_content_time.append([times_count, j + 1, i + 1, j * batches + i, loss])
                    times_count += 5
                    time_t = time.time()

            # if minideltaloss is not None and np.abs(loss - preloss) < minideltaloss:
            #    break
            # if miniloss is not None and np.abs(loss) < miniloss:
            #    break

            preloss = loss
            track_content_epoch.append([times_count, j, loss])

        tracks = pd.DataFrame(data=track_content_time, index=None,
                              columns=['time(s)', 'epoches', 'batches', 'total', 'loss'])
        tracks.to_csv('./training/training_track_{}.csv'.format(time_t))
        tracks = pd.DataFrame(data=track_content_epoch, index=None, columns=['time(s)', 'epoches', 'loss'])
        tracks.to_csv('./training/training_track_epo_{}.csv'.format(time_t))

        return loss

    def reset(self):

        for nn in self.NN:
            nn.reset()


    def predict(self, x):

        # transpose x
        x = np.asmatrix(x).T

        for layer in self.NN:
            x = layer.F(x)

        x = x.T.getA()

        return x

    def evalute(self, x, y):

        predict = self.predict(x)
        loss = self.Optimizer.loss(x, y)
        if self.Onehot:
            y = y.argmax(axis=1)
            predict = predict.argmax(axis=1)
        else:
            predict = np.round(predict)

        acc = np.mean(np.equal(y, predict))

        if self.Debug:
            print('Accuracy: {:.4f}, Total Loss: {:.4f}.'.format(acc, loss))

        return acc
