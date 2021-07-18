* [Parallel\-SGD\-docs\-zh\_CN](#parallel-sgd-docs-zh_cn)
  * [codec(编码控制器)](#codec%E7%BC%96%E7%A0%81%E6%8E%A7%E5%88%B6%E5%99%A8)
    * [codec使用样例](#codec%E4%BD%BF%E7%94%A8%E6%A0%B7%E4%BE%8B)
      * [1\. Parameter server](#1-parameter-server)
        * [1\.1 Server](#11-server)
        * [1\.2 Worker](#12-worker)
      * [2\. Peer to peer](#2-peer-to-peer)
    * [调试](#%E8%B0%83%E8%AF%95)
    * [部署](#%E9%83%A8%E7%BD%B2)
      * [1\. 部署 Worker](#1-%E9%83%A8%E7%BD%B2-worker)
      * [2\. 配置 Cluster](#2-%E9%85%8D%E7%BD%AE-cluster)
      * [3\. 提交任务](#3-%E6%8F%90%E4%BA%A4%E4%BB%BB%E5%8A%A1)
    * [API](#api)
      * [1\. BlockWeight](#1-blockweight)
    * [2\.Synchronized &amp; asynchronous](#2synchronized--asynchronous)
    * [3\. netEncapsulation](#3-netencapsulation)
    * [4\. set\_result](#4-set_result)

# Parallel-SGD-docs-zh_CN

## codec(编码控制器)
> codec是本项目的核心，通过编写不同的codec可以实现任意的网络拓扑、神经网络更新策略等等，支持同步、异步，支持传递神经网络权重、梯度，实现较为完善。

### codec使用样例

#### 1. Parameter server
> 参数服务器结构是传统机器学习的经典结构，要实现这种结构需要分别实现Client codec和Server codec，下面将简单介绍这种结构的fedavg实现。

##### 1.1 Server

参数服务器只需要实现`receive_blocks`即可，其他方法可以按照需求进行编写，不需要实现的函数使用pass略过，每当参数服务器接收到新的参数就会触发该方法，content的结构由发送者确定，这里采用典型的[0]存储发送这node_id，[1]存储网络参数的方式。

```python
import numpy as np

from numpy import ndarray
from typing import Tuple, Dict, Union, Iterable

from parallel_sgd.codec import GlobalSettings
from parallel_sgd.codec import Codec, netEncapsulation
from parallel_sgd.codec import BlockWeight
from constants import Parameter_Server


class FedAvgServer(Codec):

  def __init__(self, node_id):
    Codec.__init__(self, node_id)
    # 用于临时存储接到的参数
    self.Bak_Weights_Node: Dict[int, Union[ndarray, float]] = {}

  def dispose(self):
    self.Bak_Weights_Node.clear()

  def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
    """
        PA Server Cannot update blocks!
    :param block_weight:
    :return:
    """
    pass

  def receive_blocks(self, content: Tuple[int, ndarray]) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
    """
        PA Server receive a json_dict and send back a request
    :param content:
    :return:
    """
    # update global current state
    self.Bak_Weights_Node[content[0]] = content[1]
    # 参数服务器需要收齐才进行整体的更新，分发新模型
    if len(self.Bak_Weights_Node) == GlobalSettings.get_default().node_count:
      global_weight = np.mean(list(self.Bak_Weights_Node.values()), axis=0)
      self.dispose()
      return netEncapsulation(GlobalSettings.get_default().nodes, (Parameter_Server, global_weight))

```

##### 1.2 Worker

工作节点需要实现`update_blocks`、`receive_blocks`即可，其他方法可以按照需求进行编写，不需要实现的函数使用pass略过，`receive_blocks`可以参考上面的Server的实现，`update_blocks`用于发送自身更新的梯度，注意下面的例子因为是联邦学习，所以工作节点训练了大约2个epoch才进行发送梯度到参数服务器。

```python
from numpy import ndarray
from typing import Union, Iterable, Tuple

from parallel_sgd.codec import Codec, netEncapsulation
from parallel_sgd.codec import BlockWeight
from constants import Parameter_Server


class FedAvgClient(Codec):

  def __init__(self, node_id):
    Codec.__init__(self, node_id)
    self.__local_turn = 0
    self.__TURN = 150

  def dispose(self):
    pass

  def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
    self.__local_turn += 1
    if self.__local_turn >= self.__TURN:
      return netEncapsulation(Parameter_Server, (self.node_id, block_weight.content))
    else:
      self.set_result(block_weight.content)

  def receive_blocks(self, content: Tuple[int, ndarray]) -> None:
    self.__local_turn = 0
    self.set_result(content[1])
```
#### 2. Peer to peer
> P2P是另一种经典的分布式结构，不需要server，只需要实现worker即可。

```python
from typing import Dict, Tuple
from numpy import ndarray

from parallel_sgd.codec import GlobalSettings
from parallel_sgd.codec import BlockWeight
from parallel_sgd.codec import Codec, netEncapsulation


class Plain(Codec):

  def __init__(self, node_id):

    super().__init__(node_id)
    self.BlockWeights: Dict[int, ndarray] = dict()

  def dispose(self):
    """
        Dispose this object
    :return: None
    """
    self.BlockWeights.clear()

  def update_blocks(self, block_weight: BlockWeight) -> netEncapsulation[Tuple[int, ndarray]]:
    """
        Try collect all blocks.
    """
    self.BlockWeights[block_weight.block_id] = block_weight.content
    self.check_for_combine()
    send_to = GlobalSettings.get_default().get_adversary(block_weight.block_id)
    return netEncapsulation(send_to, (block_weight.block_id, block_weight.content))

  def receive_blocks(self, content: Tuple[int, ndarray]) -> None:
    """
        Try collect all blocks.
    """
    self.BlockWeights[content[0]] = content[1]
    self.check_for_combine()

  def check_for_combine(self):

    if len(self.BlockWeights) < GlobalSettings.get_default().block_count:
      return

    res = 0
    for val in self.BlockWeights.values():
      res += val
    self.set_result(res / len(self.BlockWeights))
    self.BlockWeights.clear()

```

### 调试

　　完成了编码控制器的编写后，我们需要对编码控制器进行 DEBUG，直接将其放入分布式集群进行测试肯定不是一个好的选择。codec.test_codec 中提供了不同类型的自动化测试脚本，在上述教程中我们编写了一个梯度平均化编码控制器，且不使用参数服务器，那么现在使用codec.test_codec.p2p_test_script.py 执行一下编码控制器的测试。  
　　找到测试脚本的第 11-22 行，用我们编写的编码控制器替换掉原有的配置，使用您的IDE进行DEBUG 或 RUN 该脚本，如果未出现错误，则证明该编码控制器在同步环境下是可用的。（注意：异步环境下的线程安全性问题比较隐蔽且难以探查，需要异步编码控制器时您应当反复检查其线程安全性，不安全的代码可能会导致意想不到的效果）  
　　假设我们的编码控制器配置在文件 codec.tutorial_codec.py 中，要修改的内容如下：

```python
# more codes upon .......

"""
    ---------------DEFINE HERE---------------
"""
# import test codec
from parallel_sgd.codec import MyComCtrl
from parallel_sgd.profiles.blockassignment.duplicate import DuplicateAssignment

# Type
SLAVE_CODEC = MyComCtrl
ASSIGNMENTS = DuplicateAssignment
"""
    ---------------DEFINE HERE---------------
"""

# more codes below ......
```

### 部署

#### 1. 部署 Worker
　　当调试完成且没有错误时，我们就可以将编码控制器部署至集群正式运行了。在可以直接互相访问的计算机上启动我们的Worker。执行以下语句：  
```shell script
python worker.py
```
#### 2. 配置 Cluster
　　记录这些Worker在同一网段内的ip地址，写入一个worker.json。假设我们的ip配置如下：
```json
{
  "PS": "192.168.1.1",
  "Worker": [
    "192.168.1.2",
    "192.168.1.3"
  ]
}
```
　　通过上述配置，我们将ip为*192.168.1.2*和*192.168.1.3*两台计算机配置为Worker节点。
　　
#### 3. 提交任务
　　将 *worker.json* 和 *job_submit.py* 放置到同一个目录，使用以下命令以默认数据集（MNIST数据集）和网络结构（Multi-Layer Perceptron）启动我们的训练集群。（假设我们新建的编码控制器在目录 *./codec/tutorial_codec.py* 中）
　　
```shell script
python job_submit.py --codec tutorial_codec.myComCtrl --node_count 2
```
　　至此，我们已经成功提交myComCtrl至集群上运行了。job_submit不会实时展示结果，要实时查看结果，可以查看worker端的控制台或worker端的log文件（在./tmp_log/目录下），当任务执行完成后，job_submit会取回log文件和训练记录csv文件，csv文件保存在根目录，log文件保存在 ./tmp_log/ 目录。  
**注意**：您需要及时收集训练信息，未收集的训练信息可能会被覆盖。  
　　执行后的输出如下所示，您也可以在 ./tmp_log/ 文件夹下找到前缀为 P-SGD Submit 的log记录。
```shell script
INFO User Submit@16:53:29 : 	 --node_count <node count 2>
INFO User Submit@16:53:29 : 	 --batch_size <batch size 64>
INFO User Submit@16:53:29 : 	 --redundancy <r 1>
INFO User Submit@16:53:29 : 	 --codec <communication codec and protocol tutorial_codec.myComCtrl>
INFO User Submit@16:53:29 : 	 --optimizer <optimizer for model training parallel_sgd>
INFO User Submit@16:53:29 : 	 --parallel_sgd <parallel_computing stochastic gradient descent synchronization type ssgd>
INFO User Submit@16:53:29 : 	 --learn_rate <learn rate for GD algorithm 0.05>
INFO User Submit@16:53:29 : 	 --epochs <training epochs 1>
INFO User Submit@16:53:29 : 	 --block_assignment <block assignment strategy iid>
INFO User Submit@16:53:29 : 	 --server_codec <parameter server codec sgq>
INFO User Submit@16:53:29 : Add worker (Rule: Worker, Id: 0, Address: 192.168.1.2).
INFO User Submit@16:53:29 : Add worker (Rule: Worker, Id: 1, Address: 192.168.1.3).
INFO User Submit@16:53:29 : Try connecting to the cluster.
INFO User Submit@16:53:31 : Connection with cluster established.
INFO User Submit@16:53:33 : Reply requirements to node(0), type(global_setting_package).
INFO User Submit@16:53:33 : Reply requirements to node(1), type(global_setting_package).
INFO User Submit@16:53:33 : Reply requirements to node(0), type(codec_and_sgd_package).
INFO User Submit@16:53:33 : Reply requirements to node(1), type(codec_and_sgd_package).
INFO User Submit@16:53:33 : Reply requirements to node(0), type(weights_and_layers_package).
INFO User Submit@16:53:33 : Reply requirements to node(1), type(weights_and_layers_package).
INFO User Submit@16:53:33 : Reply requirements to node(0), type(misc_package).
INFO User Submit@16:53:33 : Reply requirements to node(0), type(data_sample_package).
INFO User Submit@16:53:36 : Reply requirements to node(1), type(misc_package).
INFO User Submit@16:53:38 : Reply requirements to node(1), type(data_sample_package).
INFO User Submit@16:53:43 : Node(0) is ready, 2 nodes total, {0} is ready.
INFO User Submit@16:54:15 : Node(1) is ready, 2 nodes total, {0, 1} is ready.
INFO User Submit@16:54:48 : Restoring data (T-N(2)-R(1)-ID(0)-CODEC(mmmmmm).csv) from 0.
INFO User Submit@16:54:48 : Restoring data (E-N(2)-R(1)-ID(0)-CODEC(mmmmmm).csv) from 0.
INFO User Submit@16:54:48 : Restoring data (./tmp_log/Training log @ node-0_16-53-33.log) from 0.
INFO User Submit@16:54:49 : Restoring data (T-N(2)-R(1)-ID(1)-CODEC(mmmmmm).csv) from 1.
INFO User Submit@16:54:49 : Restoring data (E-N(2)-R(1)-ID(1)-CODEC(mmmmmm).csv) from 1.
INFO User Submit@16:54:49 : Restoring data (./tmp_log/Training log @ node-1_16-53-37.log) from 1.
```
### API

#### 1. BlockWeight

> BlockWeight是本项目传递消息的封装结构,其中content是一个列表，通常情况下存储该包发送的node_id,以及要发送的梯度。

```python
class BlockWeight:
    """
        Weights calculated using one block
    """

    def __init__(self, content: ndarray, block_id: int):
        self.block_id = block_id
        self.content = content
```

### 2.Synchronized & asynchronous

**注意**：在 Async-SGD 执行模式下，数据的产生与接收是异步的，update_blocks 与 receive_blocks方法可能会同时被不同的线程调用，需要额外考虑数据的线程安全性。  
**注意**：receive_blocks 方法中同样可以使用 yield netEncapsulation() 来发送数据，您可以借助这种形式实现数据包的二次加工和转发。

### 3. netEncapsulation

> 该类用来包裹要发送的数据，第一个参数用来指定发送的接收者，第二个参数用来发送自定义的数据，一般情况下封装自身node_id以及要发送的参数。

```python
def __init__(self, send_to_who: Union[Iterable[int], int], content: T):
    ...
```
### 4. set_result

> 该方法用于更新自身操作，如果是同步的调用的话，每一轮计算后都需要执行一次set_result,set_result的方法定义如下，可以看到如果里面已经有值，就直接做一个相加，如果需要其他更复杂的逻辑，可以传入一个lambda表达式。

```python
def set_result(self, content: ndarray, operation: Callable[[ndarray, ndarray], ndarray] = None):
    """
            Do some operations on current data.
        :param content: content used to modify
        :param operation: modify operation. Callable object, obtain the old result and content,
                          and returns a newer object.
                          def func(old_result: ndarray, new_content: ndarray) -> ndarray: # returns new result
        :return: None
        """
    if operation is None:
        def operation(x: ndarray, y: ndarray) -> ndarray: return x + y

        with self.__rw_lock:
            tmp = self.__updated_weight_buffer
            self.__updated_weight_buffer = operation(tmp if tmp is not None else np.asarray(0.0), content)
```

