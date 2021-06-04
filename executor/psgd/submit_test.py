import multiprocessing
import os
import shutil
import unittest

import codec.plain
import dataset
import dataset.transforms
import executor.psgd as parallel
import network
import nn
import roles

os.chdir("../../")


class TestCase(unittest.TestCase):

    def test_something(self):

        model = nn.model.SequentialModel(input_shape=[-1, 784])
        model.add(nn.layer.Dense(128, activation=nn.activation.Tanh()))
        model.add(nn.layer.Dense(128, activation=nn.activation.Tanh()))
        model.add(nn.layer.Dense(10, activation=nn.activation.Softmax()))

        model.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())

        data = dataset.MNIST()
        trans = dataset.transforms.Shuffle().add(dataset.transforms.ImageCls())

        job = parallel.ParallelSGD(model, data, trans)
        nodes = network.NodeAssignment()
        nodes.add(0, '127.0.0.1')

        worker = multiprocessing.Process(target=roles.Worker().slave_forever)
        worker.start()

        self.assertGreater(job.parallel(nodes, codec=codec.plain.Plain, epoch=2)[0]['accuracy'], 0.95)

        # clear env
        worker.terminate()
        shutil.rmtree("./Node-0-Retrieve")
        os.remove("./MODEL-P-SGD-N(0).model")
        os.remove("./TR-P-SGD-N(0).csv")
        os.remove("./EV-P-SGD-N(0).csv")


if __name__ == '__main__':
    unittest.main()
