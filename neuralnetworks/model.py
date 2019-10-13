import numpy as np
import pandas as pd

import time


class ModelFitWithMap:

    def __init__(self, nn, optimizer, onehot=True, debug=True):
        self.NN = nn
        self.Optimizer = optimizer
        self.Onehot = onehot
        self.Debug = debug

    def fit(self, x, y, epochs, batch_size):
        if batch_size > len(x):
            batch_size = len(x)

        batches = int(len(x) / batch_size)
        time_t = time.time()
        # train
        for j in range(epochs):
            for i in range(batches):
                start = i * batch_size % (len(x) - batch_size + 1)
                end = start + batch_size
                part_x = x[start:end]
                part_y = y[start:end]

                loss = self.Optimizer.train(part_x, part_y)

                interval = time.time() - time_t

                if self.Debug and interval > 1:
                    print('epochs: {}/{}, batches: {}/{}, loss: {:.4f}'.format(j + 1, epochs, i + 1, batches, loss))
                    # draw heat map of last layer, 2d only
                    w = self.NN[-1].W
                    samples = 100
                    w_1 = np.linspace(-1 + w[0,0], 1 + w[0,0], samples)
                    w_2 = np.linspace(-1 + w[0,1], 1 + w[0,1], samples)
                    loss_mech = np.zeros(shape=[samples, samples])

                    for a in range(samples):
                        for b in range(samples):
                            self.NN[-1].W = np.asarray([w_1[a], w_2[b]]).reshape(w.shape)
                            loss_mech[a][b] = self.Optimizer.loss(part_x, part_y)
                    self.NN[-1].W = w
                    grad = self.Optimizer.grad(part_x, part_y)
                    scale = 1
                    grad = -1 * grad * scale
                    import matplotlib.pyplot as plt

                    w_1, w_2 = np.meshgrid(w_1, w_2)
                    fig = plt.figure()
                    plt.contourf(w_1, w_2, loss_mech, levels=7)
                    c = plt.contour(w_1, w_2, loss_mech, colors='black')
                    plt.clabel(c, inline=True, fontsize=10)
                    plt.plot([w[0,0], w[0,0] + grad[0,0]], [w[0,1], w[0,1] + grad[0,1]], 'r-')
                    plt.plot([w[0,0] + grad[0,0]], [w[0,1] + grad[0,1]], 'r>')
                    plt.savefig('./figs/epo{}-bat{}.png'.format(j+1, i+1))
                    plt.close(fig)

                    time_t = time.time()

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

        # tracks = pd.DataFrame(data=track_content_time, index=None,
        #                       columns=['time(s)', 'epoches', 'batches', 'total', 'loss'])
        # tracks.to_csv('./training/training_track_{}.csv'.format(time_t))
        # tracks = pd.DataFrame(data=track_content_epoch, index=None, columns=['time(s)', 'epoches', 'loss'])
        # tracks.to_csv('./training/training_track_epo_{}.csv'.format(time_t))

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
        loss = self.Optimizer.loss(x, y) * 100000
        if self.Onehot:
            y = y.argmax(axis=1)
            predict = predict.argmax(axis=1)
        else:
            predict = np.round(predict)

        acc = np.mean(np.equal(y, predict))

        if self.Debug:
            print('Accuracy: {:.4f}, Total Loss: {:.4f}.'.format(acc, loss))

        return acc
