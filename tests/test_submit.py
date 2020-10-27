import time

import executor.psgd as PSGD
import nn
from codec.quantization import QuantizedClient, QuantizedParaServer
from dataset import MNIST
from dataset.transforms import ImageCls, Shuffle

if __name__ == '__main__':
    model = nn.model.Model.load('MNISTDNN.model')

    job = PSGD.ParallelSGD(model, data=MNIST(), transform=Shuffle().add(ImageCls()))
    for i in range(1, 11):
        try:
            nodes = PSGD.parse_worker(worker_cnt=i, ps=False)
            job.parallel(nodes, codec=QuantizedClient,
                         epoch=80,
                         op_type=nn.optimizer.GradientAveragingOptimizer,
                         ps_codec=QuantizedParaServer,
                         gd_type=nn.gradient_descent.SGDOptimizer,
                         gd_params=(0.005,),
                         mission_title="FP[{}nodes]".format(i),
                         ssgd_timeout_limit=1000)
        except ConnectionAbortedError:
            print("Worker exited without reports.")
        time.sleep(10)
