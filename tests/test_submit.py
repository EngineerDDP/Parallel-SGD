import executor.psgd as PSGD
import nn
from codec.plain import Plain
from dataset import MNIST, CIFAR, SimLin
from dataset.transforms import ImageCls, Shuffle

if __name__ == '__main__':
    model = nn.model.Model.load('abc.model')

    job = PSGD.ParallelSGD(model, data=CIFAR(), transform=Shuffle().add(ImageCls()))
    nodes = PSGD.parse_worker(worker_cnt=1, ps=False)
    for i in range(1):
        try:
            job.parallel(nodes, codec=Plain, epoch=1, op_type=nn.optimizer.PSGDOptimizer,
                         gd_type=nn.gradient_descent.ADAMOptimizer)
        except ConnectionAbortedError:
            print("Worker exited without reports.")
