import nn
import executor.psgd as parallel

from codec.naive_ps import PSClient, GradDiffPS
from codec.adam_gradiff import ADAMParameterServer
from dataset import CIFAR
from dataset.transforms import ImageCls, Shuffle

model = nn.model.SequentialModel(input_shape=[-1, 32, 32, 3])
model.add(nn.layer.Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
model.add(nn.layer.Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
model.add(nn.layer.Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
model.add(nn.layer.Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
model.add(nn.layer.Flatten())
model.add(nn.layer.Dropout())
model.add(nn.layer.Dense(units=128, activation=nn.activation.Tanh()))
model.add(nn.layer.Dropout())
model.add(nn.layer.Dense(units=10, activation=nn.activation.Softmax()))

model.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())

data = CIFAR()
trans = Shuffle().add(ImageCls())

job = parallel.ParallelSGD(model, data, trans)
nodes = parallel.parse_worker(worker_cnt=10, ps=True)

print(job.parallel(nodes, codec=PSClient,
             epoch=80,
             op_type=nn.optimizer.PSGDOptimizer,
             ps_codec=ADAMParameterServer,
             gd_type=nn.gradient_descent.SGDOptimizer,
             gd_params=(1.0,),
             mission_title="FPTest-CIFAR",
             ssgd_timeout_limit=20000))
