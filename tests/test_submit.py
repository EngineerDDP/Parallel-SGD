import time

import executor.psgd as PSGD
import nn
from codec.tqn import TQNClient, TQNServer
from dataset import MNIST
from dataset.transforms import ImageCls, Shuffle

if __name__ == '__main__':
    model = nn.model.Model.load('MNISTDNN.model')

    job = PSGD.ParallelSGD(model, data=MNIST(), transform=Shuffle().add(ImageCls()))
    for i in range(1, 11):
        try:
            nodes = PSGD.parse_worker(worker_cnt=i, ps=True)
            job.parallel(nodes, codec=TQNClient,
                         epoch=80,
                         op_type=nn.optimizer.GradientAveragingOptimizer,
                         ps_codec=TQNServer,
                         gd_type=nn.gradient_descent.SGDOptimizer,
                         gd_params=(0.005,),
                         mission_title="FP[{}nodes]".format(i))
        except ConnectionAbortedError:
            print("Worker exited without reports.")
        time.sleep(10)
