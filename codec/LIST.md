# 内置编码器列表

## 概览

### Worker 控制器

|索引字符串|引用路径|类别|简介|
|----|----|----|----|
|"ccdc"|codec.ccdc.CodedCommunicationCtrl|P2P|基于分布式编码的分布式梯度下降算法|
|"plain"|codec.plain.PlainCommunicationCtrl|P2P|全连接且密集通讯的分布式梯度下降算法|
|"ps"|codec.naive_ps.PAClientCodec|PS|简单的基于参数服务器交换的并行梯度下降算法|
|"ndc"|codec.ndc.NaiveDuplicationCodec|P2P|基于简单复制拷贝实现降低通信负载的分布式梯度下降算法|
|"unicast"|codec.unicast.UnicastCommunicationCtrl|P2P|用于在参数服务器环境下模拟单播通讯，现已废弃|
|"q1"|codec.quantization.Quantization1BitPSCodec|PS|确定二值化参数平均化方法|
|"q2"|codec.quantization.Quantization2BitPSCodec|PS|确定三值化参数平均化方法|
|"sgq"|codec.sgq.SGQClient|PS|基于随机三值化和哈夫曼编码的异步梯度下降更新策略|

### Server 控制器

|索引字符串|引用路径|简介|
|----|----|----|
|"simple"|codec.naive_ps.ParaServerCodec|参数平均化方法|
|"grad"|codec.naive_ps.GradDiffParaServerCodec|梯度增量更新策略|
|"dc"|codec.dc_asgdcodec.DCASGDServerCodec|带延迟补偿的梯度增量更新策略|
|"sgq"|codec.sgq.SGQServer|基于哈夫曼编码和随机三值化的增量更新策略|
|"qpa"|codec.quantization.FPWParaServer|全精度量化梯度平均化处理策略|

## 介绍

### ccdc

　　基于分布式编码的同步的并行梯度下降算法控制器，是一种基于P2P Reduce策略优化的并行算法。
### plain

　　全连接密集传输的同步的并行梯度下降算法，每个节点都将其梯度广播给不拥有该信息的所有节点，是一种朴素的
P2P同步更新策略。当没有冗余时，总通信量为 $O(n^2)$ 级别。

### ps

　　朴素的梯度平均化方法，Codec将本地数据上传到参数服务器，等待返回值后直接交付Optimizer执行。
可用的调用组合如下所示：  
1.朴素的带参数服务器的异步梯度下降算法，每个节点上传计算所得的梯度，参数服务器以异步的形式平均梯度，
并返回梯度增量值。
```shell script
python job_submit.py --codec ps --server_codec grad --optimizer psgd
```
2.朴素的参数平均化方法，每个节点上传计算所得梯度，参数服务器以异步的形式平均梯度，并返回由参数服务器
控制更新的参数，节点接收到参数后覆盖原参数。参考文献：  
*Recht, Benjamin and Re, Christopher and Wright, Stephen and Feng Niu. Hogwild: A Lock-Free 
Approach to Parallelizing Stochastic Gradient Descent. Advances in Neural Information Processing 
Systems (NIPS). Curran Associates, Inc. 2011.*
。
```shell script
python job_submit.py --codec ps --server_codec simple optimizer pa
```
3.带延迟补偿的梯度增量更新方法。是一种基于二阶梯度的并行优化方法，参考文献：  
*Shuxin Zheng, Qi Meng, Taifeng Wang, et al. Asynchronous Stochastic Gradient Descent with Delay 
Compensation. International Conference on Machine Learning (ICML), Sydney, Australia. 2017.*
```shell script
python job_submit.py --codec ps --server_codec dc --optimizer pa
```
4.基于双重缓冲的异步并行梯度下降算法。关于双重缓冲的执行流程，请参考文献：    
*__call__. Seide, H. Fu, J. Droppo, et al. 1-Bit stochastic gradient descent and its 
application to data-parallel distributed training of speech DNNs \[C\]// 15th Annual 
Conference of the International Speech Communication Association (INTERSPEECH), 
Singapore, 2014:1058-1062.*。
```shell script
python job_submit.py --codec ps --server_codec grad --optimizer fast 
```
### ndc

　　朴素的基于复制备份的并行梯度下降算法，每份数据被复制拷贝发送到不同的节点上。稍加修改就可以实现
分布式容灾。  
（提示：借助SSGD超时补救机制，SSGD超时后会自动调用Codec的do_something_to_save_yourself()方法）

### q1 & q2

　　朴素的确定量化方法（Deterministic Quantization Method），使用sign函数执行二值化量化，借助
标准差界实现三值化量化。一般和*qpa*组合使用。有以下用法：  
1.层同步更新的量化并行梯度下降算法。
```shell script
python job_submit.py --codec q1 --server_codec qpa --optimizer pa
```

### sgq 
　　待验证。核心编码器使用C++封装。