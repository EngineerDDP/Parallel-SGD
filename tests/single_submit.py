import executor.psgd as PSGD
import nn
from codec.plain import Plain
from dataset import MNIST
from dataset.transforms import ImageCls, Shuffle

if __name__ == '__main__':
    model = nn.model.Model.load('MNISTDNN.model')

    job = PSGD.ParallelSGD(model, data=MNIST(), transform=Shuffle().add(ImageCls()))
    nodes = PSGD.parse_worker(worker_cnt=1, ps=False)
    job.parallel(nodes, codec=Plain, epoch=1)
