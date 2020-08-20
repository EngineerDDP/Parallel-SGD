# Parallel SGD

　　Parallel SGD已经更新到r0.4版本了，使用了全新的网络与调度架构重新设计，再也不需要通过复杂的
调度脚本去挨个节点拉取 log 文件了，所有的参数使用 job_submit 一次提交到处运（崩）行（溃）。  
　　本项目用于联邦学习和分布式学习中的关于网络架构和通信编码部分的实验框架，通过本项目暴漏出的ICommunication_Ctrl
接口（详情参阅  [编码控制器教程](./codec/README.md) ），可以实现选择性的建立节点之间的通讯机制，并对数据网络上
流通的数据进行加解密等操作。当您需要对网络拓扑进行改动时，仅需要修改ICommunication_Ctrl模块来改变网络通信的目的和
方式，底层自建的TCP长连接会帮助您将数据包送达至指定目的地。

## 参数说明

### 工作节点参数
　　所有的参数都通过 job_submit.py 传入，worker节点无需传入任何参数。启动时，使用以下命令启动Worker，无需传入参数。
当任务提交时，节点会自动申请并获取工作状态信息。  
**注意**：每个worker所在的计算机都需要允许15387端口的TCP传入。
```shell script
python worker.py 
```

### 任务提交
　　提交任务到集群时，使用 job_submit.py 脚本，脚本参数声明如下：
```shell script
python job_submit.py 
    --node-count 4  
    --batch-size 128  
    --redundancy 1  
    --codec plain,plain,plain,plain,ccdc,ps  
    --optimizer psgd  
    --psgd ssgd  
    --learn-rate 0.05  
    --epochs 10  
    --dataset mnist
    --non-iid 
    --block-assignment iid 
    --server-codec grad 
    --workers worker.json
```
* *-n* *node-count*  
节点数目，当前任务需要占用的节点数目。  
该参数需要与代码适配，在 GlobalSetting 中默认的每个节点编号是连续的且从0开始。

* *-b* *batch_size*  
批次大小，当前任务训练批次的大小。  
**注意**：批次在每个节点上是均分的，当冗余设置为 1 倍时，每个节点上的训练样本数目为
*batch_size* / *node_count*。

* *-r* *redundancy*  
样本冗余份数，当前任务所需的样本冗余份数。  
样本冗余份数会按要求传送给 GlobalSetting ，具体的冗余分配方案仍然依赖 *block_assignment* 
参数，当 *block_assignment* 参数提供的处理方法无法处理冗余分配时，设置的冗余级别事实上是无效的。  
**注意**：如果编码控制器无法处理冗余分配情况，可能会导致全局死锁。

* *-O* *optimizer*  
使用的梯度下降优化器。  
选用并行梯度下降优化器以实现并行计算，选用单机梯度下降优化器只能执行单机计算。  
（关于可用的梯度下降优化器，请参阅 [梯度下降优化器类型](./nn/LIST.md) ）

* *-C* *codec*  
worker上执行的实际编码器，当需要与参数服务器协同工作时，该编码器要能与参数服务器上执行的编码器匹配。
当传入一个编码器参数时，默认给每层都分配相同的编码器。需要传入多个编码器时，使用逗号隔开，每个编码器
对应一层，确保层数和编码器数量一致。 
编码器类继承自 codec.interfaces.ICommunicationCtrl 实现一个编码器类并在 server_util.init_model.__codec_map 
中注册，即可在此处传入对应参数，启动对应的客户端编码器。   
**注意**：第一个编码器参数对应第一个层，以此类推。  
（关于编码器设计的详情，请参阅 [编码控制器教程](./codec/README.md) ）  
（关于可用的已实现的编码器，请参阅 [编码器类型](./codec/LIST.md) ）  

* *psgd*  
worker上执行的实际SGD同步器。  
asgd 对应异步梯度下降算法，执行异步更新策略，非阻塞立即返回能够获取到的最新权重；ssgd 对应同步梯度下降算法，执行
同步更新策略，当且仅当已经和必要节点执行完参数同步后才会释放锁，继续进行训练，ssgd 同样有保底重传策略，当超出
SGD 最长同步等待时间后，ssgd 会调用每一层编码器的 ICommunicationCtrl.do_something_to_save_yourself 方法，
尝试补救，当两次超时并且无法挽回后，ssgd 会报告超时错误。  

* *learn_rate*  
worker上执行的学习率，当受参数服务器控制更新时，此参数相当于无效。

* *-E* *epochs*  
worker上执行的训练轮次。

* *-D* *dataset*  
训练所使用的数据集，目前内置有 **MNIST** 数据集，**CIFAR-10** 数据集。  

* *--non-iid*  
加入此选项，使用非i.i.d.数据集划分。  

* *block_assignment*  
全局训练样本分配策略。  
继承自 profiles.blockassignment.interfaces.IBlockAssignment ，使用自定义的 block_assignment 分配样本，需要
在 server_util.init_model.__assignment_map 中注册。
本项目的样本被划分为训练集与测试集，样本是固定的。训练集又被划分为多个batch，每个batch被均分为多个block，并发送到
block_assignment 指定的节点上。需要划分为多少个block，以及每个block复制多少份发送给多少节点由block_assignment决定。  
（关于分配策略的详情，请参阅 [分配策略](./profiles/blockassignment/README.md) ）

* *server_codec*  
参数服务器编码器。  
继承自 codec.interfaces.ICommunicationCtrl ，实现一个编码器并在 server_util.init_model.__para_server_map 中
注册，即可在此处传入对应参数，启动对应的参数服务器编码器。

* *workers*  
工作节点目录，参数内容为文件名，默认为 worker.json。  
在提交任务到集群上之前，需要设置worker.json，标明当前集群的节点列表。
worker.json格式如下：
```json
[
    ["PS", "192.168.1.2"], 
    ["Worker", "192.168.1.3"]
]
```
　　主体为一个数组，每行包含两个信息，分别是该节点的工作角色和IP地址，您要保证这些IP地址均可以互相访问。
目前支持的角色类型只有两种，"PS"代表该节点以参数服务器的形式工作，"Worker"代表该节点以计算
节点的形式工作。   
**注意**：无需在每个节点上配置worker.json，只需要在提交任务时配置了worker.json即可。  

### 工作与等待

　　根据控制台的输出，您可以确定已经成功提交任务至多少个节点，当所有节点都准备就绪时，您提交的任务就在集群上正常运行。
当一个Worker已经初始化完成后，会输出相应的信息，以及总共需要初始化的Worker数目，输出如下。
```shell script
INFO Coordinator-192.168.1.1@10:49:55 : Add worker (Rule: PS, Id: -2, Address: 192.168.1.2).
INFO Coordinator-192.168.1.1@10:58:38 : Add worker (Rule: Worker, Id: 0, Address: 192.168.1.3).
INFO Coordinator-192.168.1.1@10:49:55 : Try connecting to the cluster.
INFO Coordinator-192.168.1.1@10:49:55 : Connection with cluster established.
INFO Coordinator-192.168.1.1@10:49:57 : Reply requirements to node(-2), type(global_setting_package).
INFO Coordinator-192.168.1.1@10:49:57 : Reply requirements to node(-2), type(codec_and_sgd_package).
INFO Coordinator-192.168.1.1@10:49:57 : Reply requirements to node(-2), type(weights_and_layers_package).
INFO Coordinator-192.168.1.1@10:49:57 : Reply requirements to node(-2), type(misc_package).
INFO Coordinator-192.168.1.1@10:49:57 : Node(-2) is ready, 2 nodes total, {-2} is ready.
INFO Coordinator-192.168.1.1@10:50:39 : Reply requirements to node(0), type(global_setting_package).
INFO Coordinator-192.168.1.1@10:50:39 : Reply requirements to node(0), type(codec_and_sgd_package).
INFO Coordinator-192.168.1.1@10:50:39 : Reply requirements to node(0), type(weights_and_layers_package).
INFO Coordinator-192.168.1.1@10:50:40 : Reply requirements to node(0), type(misc_package).
INFO Coordinator-192.168.1.1@10:50:40 : Reply requirements to node(0), type(data_sample_package).
INFO Coordinator-192.168.1.1@10:50:44 : Node(0) is ready, 2 nodes total, {-2, 0} is ready.
```
　　此时您可以选择通过按下 Ctrl+C 键手动退出 job_submit，也可以选择等待所有Worker返回数据集给您。当您选择提前退出job_submit
时，您需要在任务运行完成之后通过以下命令从每个节点上回收上次执行的训练数据。
```shell script
python job_submit.py --retrieve_data --worker ./worker.json
```
　　连接无误的话，输出应当如下所示：
```shell script
INFO Coordinator-192.168.1.1@11:12:26 : Add worker (Rule: Worker, Id: 0, Address: 192.168.1.3).
INFO Coordinator-192.168.1.1@11:12:26 : Add worker (Rule: PS, Id: -2, Address: 192.168.1.2).
INFO Coordinator-192.168.1.1@11:12:26 : Try connecting to the cluster.
INFO Coordinator-192.168.1.1@11:12:26 : Connection with cluster established.
INFO Coordinator-192.168.1.1@11:12:27 : Acquire log file from worker(0).
INFO Coordinator-192.168.1.1@11:12:27 : Acquire log file from worker(-2).
INFO Coordinator-192.168.1.1@11:12:27 : Save log file for worker(0).
INFO Coordinator-192.168.1.1@11:12:27 : Save log file for worker(0).
INFO Coordinator-192.168.1.1@11:12:27 : Save log file for worker(0).
INFO Coordinator-192.168.1.1@11:12:27 : Save log file for worker(0).
INFO Coordinator-192.168.1.1@11:12:27 : Save log file for worker(-2).
INFO Coordinator-192.168.1.1@11:12:27 : Save log file for worker(-2).
```
**注意**：.log 文件在训练阶段就可以给出，.csv 报表要在全部训练过程结束之后才能给出。预估您任务的执行时间，来获得完整的数据。  
**注意**：参数服务器只有Worker工作记录和简要的Training日志，没有详细的训练过程报表。  
**注意**：在Sync-SGD强一致性约束下，集群可以保证每个Worker给出的Evaluation报表是一致的。  
**注意**：在ASync-SGD弱一致性约束下，每个Worker给出一个有限的感知范围下最优的Evaluation报表。  

### 提交流程

　　训练任务的提交过程遵循分布式系统的三阶段提交规范。包含canCommit、preCommit和doCommit三个阶段。
* canCommit阶段：Coordinator会逐个访问worker.json中的每个Worker，检查Worker是否处于可提交阶段。  
    - 当所有的Worker都接收了本次连接并回复ACK之后，进入preCommit阶段。
    - 当存在一个Worker处于Busy状态，或存在一个Worker连接超时，取消提交。
* preCommit阶段：Coordinator会向每个Worker提交本次训练所需的超参数、初始化权重、网络结构和样本集等数据。
    - Worker逐个接收并确认每个Package，当所有的Worker都完成确认之后，进入doCommit阶段。
    - 当存在一个Worker未确认，或超时未确认之后，Coordinator就会放弃提交，断开当前连接。
* doCommit阶段：第一个完成preCommit的Worker会向集群中广播Ready标志。
    - 每个完成preCommit阶段的Worker都会在收到Ready标志后回复Ready_Type，当所有Worker都进入Ready状态后，提交完成。
    - 当超时未收到指定数目的Ready回复后，所有的Worker回退到初始状态并重置连接状态，Coordinator检查连接断开后报告任务提交失败。
    
### 一致性约束

　　在 Sync-SGD Type 约束下，网络权重参数满足一致性（Consistency）和分区容错性（Partition Tolerance），不满足可用性（Availability）。
详细的资源CAP状况如下表。
 
|资源|Consistency|Availability|Partition Tolerance|
|----|----|----|----|
|初始化资源|√|√|×|
|训练日志|√|√|×|
|评估日志|√|×|√|
|网络权重|√|×|√|

　　在 ASync-SGD Type 约束下，网络权重参数满足分区容错性（Partition Tolerance）和可用性（Availability），不满足一致性（Consistency）。
详细的资源CAP状况如下表。

|资源|Consistency|Availability|Partition Tolerance|
|----|----|----|----|
|初始化资源|√|√|×|
|训练日志|√|√|×|
|评估日志|×|√|√|
|网络权重|×|√|√|

## 框架结构

　　To be constructed.