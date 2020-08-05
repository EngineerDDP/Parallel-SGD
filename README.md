# Parallel SGD

　　Parallel SGD已经更新到r0.4版本了，使用了全新的网络与调度架构重新设计，再也不需要通过复杂的
调度脚本去挨个节点拉取 log 文件了，所有的参数使用 job_submit 一次提交到处运（崩）行（溃）。
此外，r0.4重构了原有的 network.agreements ，从 agreements 开刀，全面消减模块之间的互相依赖
关系，大幅精简了调用复杂度。  
　　本段没内容了，往下看使用说明。

## 参数说明

### 工作节点参数
　　所有的参数都通过 job_submit.py 传入，worker节点无需传入任何参数。启动时，使用以下命令启动Worker，无需传入参数。当任务提交时，节点会自动申请并获取工作状态信息。
```shell script
python worker.py 
```
　　在提交任务到集群上之前，需要在任意节点设置worker.json，标明当前集群的节点列表。
worker.json格式如下：
```json
[
    [0, "ce8f76ef", "192.168.1.1"], 
    [1, "a07f9e18", "192.168.1.2"]
]
```
　　主体为一个数组，每行包含一个可用的worker信息，第一列是为worker分配的id，
第二列是用于识别改任务类别的worker uuid，第三列是用于连接的网络地址。  
**注意**：无需在每个节点上配置worker.json，只需要在其中一个节点上配置了worker.json即可。  

### 任务提交
　　提交任务到集群时，使用 job_submit.py 脚本，脚本参数声明如下：
```shell script
python job_submit.py 
    --node_count 4  
    --batch_size 128  
    --redundancy 2  
    --codec ccdc  
    --psgd ssgd  
    --learn_rate 0.05  
    --epochs 10  
    --block_assignment iid 
    --server_codec graddiff 
    --workers worker.json
```
* *node_count*  
节点数目，当前任务需要占用的节点数目。  
该参数需要与代码适配，在 GlobalSetting 中默认的每个节点编号是连续的且从0开始。

* *batch_size*  
批次大小，当前任务训练批次的大小。  
**注意**：批次在每个节点上是均分的，当冗余设置为 1 倍时，每个节点上的训练样本数目为
*batch_size* / *node_count*。

* *redundancy*  
样本冗余份数，当前任务所需的样本冗余份数。  
样本冗余份数会按要求传送给 GlobalSetting ，具体的冗余分配方案仍然依赖 *block_assignment* 
参数，当 *block_assignment* 参数提供的处理方法无法处理冗余分配时，设置的冗余级别事实上是无效的。  
**注意**：如果编码控制器无法处理冗余分配情况，可能会导致全局死锁。

* *codec*  
worker上执行的实际编码器，当需要与参数服务器协同工作时，该编码器要能与参数服务器上执行的编码器匹配。  
编码器类继承自 codec.interfaces.ICommunicationCtrl 实现一个编码器类并在 server_util.init_model.__codec_map 
中注册，即可在此处传入对应参数，启动对应的客户端编码器。  
（关于编码器设计的详情，请参阅[编码控制器教程](./codec/README.md)）

* *psgd*  
worker上执行的实际SGD同步器。  
asgd 对应异步梯度下降算法，执行异步更新策略，非阻塞立即返回能够获取到的最新权重；ssgd 对应同步梯度下降算法，执行
同步更新策略，当且仅当已经和必要节点执行完参数同步后才会释放锁，继续进行训练，ssgd 同样有保底重传策略，当超出
SGD 最长同步等待时间后，ssgd 会调用每一层编码器的 ICommunicationCtrl.do_something_to_save_yourself 方法，
尝试补救，当两次超时并且无法挽回后，ssgd 会报告超时错误。  

* *learn_rate*  
worker上执行的学习率，当受参数服务器控制更新时，此参数相当于无效。

* *epochs*  
worker上执行的训练轮次。

* *block_assignment*  
全局训练样本分配策略。  
继承自 profiles.blockassignment.interfaces.IBlockAssignment ，使用自定义的 block_assignment 分配样本，需要
在 server_util.init_model.__assignment_map 中注册。
本项目的样本被划分为训练集与测试集，样本是固定的。训练集又被划分为多个batch，每个batch被均分为多个block，并发送到
block_assignment 指定的节点上。需要划分为多少个block，以及每个block复制多少份发送给多少节点由block_assignment决定。  
（关于分配策略的详情，请参阅[分配策略](./profiles/blockassignment/README.md)）

* *server_codec*  
参数服务器编码器。  
继承自 codec.interfaces.ICommunicationCtrl ，实现一个编码器并在 server_util.init_model.__para_server_map 中
注册，即可在此处传入对应参数，启动对应的参数服务器编码器。

* *workers*  
工作节点目录，参数内容为文件名，默认为 worker.json。

## 框架结构

pass