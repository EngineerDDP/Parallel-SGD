import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    SIM_ACC = 1000
    DRAW_ACC = 200

    x1 = np.linspace(1, 1, SIM_ACC)
    x2 = np.linspace(5, 5, SIM_ACC)
    dw = np.ones_like(x2) * 5
    dw[:200] *= -1
    y = np.asarray([x1 * dw, x2 * 0])

    y_ = y + np.random.uniform(low=-1, high=1, size=y.shape)

    def evaluate(p1, p2):
        return np.mean(np.square(y_ - (np.asarray([x1 * p1, x2 * p2]))))

    def evaluate_rand(p1, p2):
        if p1 > 0:
            sample = np.random.choice(np.arange(0, SIM_ACC, 1), size=1)
        else:
            sample = np.random.choice(np.arange(0, SIM_ACC, 1), size=1)
        y_s = y_[:,sample]
        x1_s = x1[sample]
        x2_s = x2[sample]

        return np.mean(np.square(y_s - (np.asarray([x1_s * p1, x2_s * p2]))))

    w1 = np.linspace(-10, 10, DRAW_ACC)
    w2 = np.linspace(-2, 2, DRAW_ACC)

    loss = np.zeros(shape=[DRAW_ACC, DRAW_ACC])
    for i in range(DRAW_ACC):
        for j in range(DRAW_ACC):
            loss[j][i] = evaluate_rand(w1[i], w2[j])

    w1_, w2_ = np.meshgrid(w1, w2)

    plt.figure(figsize=(8, 4))
    plt.contourf(w1_, w2_, loss, 10)
    cset = plt.contourf(w1_, w2_, loss, 10)
    # contour = plt.contour(w1_, w2_, loss, 10, colors='k')
    # plt.clabel(contour, fontsize=10, colors='k')
    plt.colorbar(cset)
    plt.show()
