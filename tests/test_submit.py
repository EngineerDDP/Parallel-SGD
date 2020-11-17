import time

import executor.psgd as PSGD
import nn
from codec.q_ext_ext import DC_QSGDClient, DC_QSGDServer
from dataset import MNIST
from dataset.transforms import ImageCls, Shuffle

if __name__ == '__main__':
    model = nn.model.Model.load('MNISTDNN.model')

    job = PSGD.ParallelSGD(model, data=MNIST(), transform=Shuffle().add(ImageCls()))
    for i in range(1, 11):
        try:
            nodes = PSGD.parse_worker(worker_cnt=i, ps=True)
            job.parallel(nodes, codec=DC_QSGDClient,
                         epoch=80,
                         op_type=nn.optimizer.PSGDOptimizer,
                         ps_codec=DC_QSGDServer,
                         gd_type=nn.gradient_descent.SGDOptimizer,
                         gd_params=(0.005,),
                         mission_title="DCQSGD[{}nodes]".format(i),
                         ssgd_timeout_limit=10000)
        except ConnectionAbortedError:
            print("Worker exited without reports.")
        time.sleep(10)
