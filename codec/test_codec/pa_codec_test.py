import numpy as np

from codec.pacodec import PAServerCodec, PAClientComPack

from server_util.init_model import ModelMNIST

from neuralnetworks.optimizer import GradientDecentOptimizer

from settings import GlobalSettings

from log import Logger


class Test_PAServer(PAServerCodec):

    def __init__(self, node_id, logger=Logger('Test')):

        PAServerCodec.__init__(self, node_id, logger)

        self.Working_Batch = [0 for node in GlobalSettings.getDefault().Nodes]
        self.W_copy = np.array(ModelMNIST.Neural_Network[0].W)

    def receive_blocks(self, json_dict):

        compack = PAClientComPack.decompose_compack(json_dict)

        self.Working_Batch[compack.Node_ID] += 1
        if compack.Layer_ID == 0 and json_dict['NW_Type'] == 'w' and self.Working_Batch[compack.Node_ID] % 10 == 0:
            self.run_test_method(compack.Content)
        if json_dict['NW_Type'] == 'b':
            ModelMNIST.Neural_Network[0].B = -1 * self.Learn_Rate * self.Current_Weights / GlobalSettings.getDefault().Batch.Batch_Size

        return super().receive_blocks(json_dict)

    def run_test_method(self, content):

        x, y = ModelMNIST.eval_data()
        nn = ModelMNIST.Neural_Network
        loss = ModelMNIST.loss_type()()
        op = GradientDecentOptimizer(loss, nn)
        w = self.W_copy - self.Learn_Rate * self.Current_Weights / GlobalSettings.getDefault().Batch.Batch_Size

        samples = 100
        w_1 = np.linspace(-1 + w[0,0], 1 + w[0,0], samples)
        w_2 = np.linspace(-1 + w[0,1], 1 + w[0,1], samples)
        loss_mech = np.zeros(shape=[samples, samples])

        for a in range(samples):
            for b in range(samples):
                nn[-1].W = np.asarray([w_1[a], w_2[b]]).reshape(w.shape)
                loss_mech[a][b] = op.loss(x, y)

        grad = content

        scale = 1
        grad = -1 * grad * scale

        import matplotlib.pyplot as plt

        w_1, w_2 = np.meshgrid(w_1, w_2)
        fig = plt.figure()
        plt.contourf(w_1, w_2, loss_mech, levels=7)
        c = plt.contour(w_1, w_2, loss_mech, colors='black')
        plt.clabel(c, inline=True, fontsize=10)
        plt.plot([w[0, 0], w[0, 0] + grad[0, 0]], [w[0, 1], w[0, 1] + grad[0, 1]], 'r-')
        plt.plot([w[0, 0] + grad[0, 0]], [w[0, 1] + grad[0, 1]], 'r>')
        plt.savefig('./figs/bat{}.png'.format(np.max(self.Working_Batch) + 1))
        plt.close(fig)

