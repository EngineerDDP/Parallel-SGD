import multiprocessing
import os
import shutil
import time
import unittest

import rpc

import nn.dataset
import nn.dataset.transforms

import parallel_sgd as parallel
import parallel_sgd.codec.plain

import network
import nn

os.chdir("../")


class TestCase(unittest.TestCase):

    def test_something(self):
        try:
            model = nn.model.SequentialModel(input_shape=[-1, 784])
            model.add(nn.layer.Dense(128, activation=nn.activation.Tanh()))
            model.add(nn.layer.Dense(128, activation=nn.activation.Tanh()))
            model.add(nn.layer.Dense(10, activation=nn.activation.Softmax()))

            model.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())

            data = nn.dataset.MNIST()
            trans = nn.dataset.transforms.Shuffle().add(nn.dataset.transforms.ImageCls())

            job = parallel.ParallelSGD(model, data, trans)
            nodes = network.NodeAssignment()
            nodes.add(0, '127.0.0.1')

            worker = multiprocessing.Process(target=rpc.Cohort().slave_forever)
            worker.start()

            self.assertGreater(job.parallel(nodes, codec=parallel_sgd.codec.plain.Plain, epoch=2)['accuracy'], 0.95)

            worker.kill()

            time.sleep(1)
        finally:
            # clear env
            shutil.rmtree("./Node-0-Retrieve")
            shutil.rmtree("./tmp_log")
            os.remove("./MODEL-P-SGD-N(0).model")
            os.remove("./TR-P-SGD-N(0).csv")
            os.remove("./EV-P-SGD-N(0).csv")


if __name__ == '__main__':
    unittest.main()
