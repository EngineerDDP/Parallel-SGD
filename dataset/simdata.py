import numpy as np
import matplotlib.pyplot as plt

w_1 = np.linspace(-1, 1, 100)
w_2 = np.linspace(-1, 1, 100)

x_1 = np.linspace(-1, 1, 100)
x_2 = np.linspace(-1, 1.5, 100)
y = 0.2*np.sin(x_1) + 0.7*np.cos(x_2)
y = y + np.random.normal(scale=0.1, size=x_1.shape)


# def loss_sim(w_1, w_2):
#     w = np.asarray([w_1, w_2])
#     w = w.reshape([1,2])
#     x = np.asarray([x_1, x_2])
#     # x = x.transpose()
#     y_l = y.reshape([1,100])
#     ch = np.random.choice(100, 10)
#     loss_r = 0
#     for c in ch:
#         loss_r += np.mean(np.square(np.tanh(np.dot(w, x[:, c]) - y_l[:, c])))
#     return loss_r / len(ch)
#
#
# l = np.zeros([len(w_1),len(w_2)])
# for i in range(len(w_1)):
#     for j in range(len(w_2)):
#         l[i][j] = loss_sim(w_1[i], w_2[j])
#
# w_1_p, w_2_p = np.meshgrid(w_1, w_2)
# plt.contourf(w_1_p, w_2_p, l, levels=7)
# c = plt.contour(w_1_p, w_2_p, l, colors='black')
# plt.clabel(c, inline=True, fontsize=10)
# plt.show()

from neuralnetworks.activations import Linear, Tanh
from neuralnetworks.layers import FCLayer
from neuralnetworks.model import Model, ModelFitWithMap
from neuralnetworks.losses import MseLoss
from neuralnetworks.optimizer import GradientDecentOptimizer

nn = [
    FCLayer(units=2, act=Tanh()),
    FCLayer(units=1, act=Tanh())
]
loss = MseLoss()
op = GradientDecentOptimizer(loss, nn, 0.1)
model = ModelFitWithMap(nn, op, False)
x = np.asarray([x_1, x_2])
model.fit(x.transpose(), y.reshape([-1, 1]), 100000, 100)

pred = model.predict(x.transpose())
pred = pred.reshape(-1)

plt.plot(x_1, y, 'b-')
plt.plot(x_1, pred, 'g.')
plt.show()

print(nn[0].W)
print(nn[0].B)

