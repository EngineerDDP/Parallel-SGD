import nn
import executor.psgd as parallel

from codec.plain import Plain
from dataset import MNIST
from dataset.transforms import ImageCls, Shuffle

model = nn.model.SequentialModel(input_shape=[-1, 784])
model.add(nn.layer.Dense(128, activation=nn.activation.Tanh()))
model.add(nn.layer.Dense(128, activation=nn.activation.Tanh()))
model.add(nn.layer.Dense(10, activation=nn.activation.Softmax()))

model.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())

data = MNIST()  # 使用MNIST数据集
trans = Shuffle().add(ImageCls())  # 先对数据集做Shuffle操作，再对数据集进行像素分类处理

job = parallel.ParallelSGD(model, data, trans)
nodes = parallel.parse_worker(worker_cnt=1)

# 两轮训练后，大约可以给出 'Loss': 0.262, 'accuracy': 0.954 的结果
print(job.parallel(nodes, codec=Plain, epoch=2))
