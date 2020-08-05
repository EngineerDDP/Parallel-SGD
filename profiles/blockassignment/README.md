# 分配策略

　　数据分配策略是 P-SGD 的另一个核心执行流程，在 Worker 初始化阶段，job_submit 就会使用对应的
数据分配策略创建数据分配集，并分发给每个节点执行计算过程。

## 接口

　　数据分配策略类实现以下接口：  
```python
from profiles.blockassignment.interfaces import IBlockAssignment
```

　　要实现一个数据分配策略，继承上述接口并实现其中的四个属性。数据分配策略由 GlobalSettings 创建
创建时传入 node_count 和 redundancy 两个参数，分别指示了当前节点数目和当前配置的冗余情况。  
　　我们如果要创建一个分配策略，使得每个 batch 划分为 node_count 份 block，每个 block 在每个对应
的节点上重复 redundancy 份。假设当前我们有4个节点，那么样本会被划分为4份，设置 redundancy=2 那么当前
每个节点上的数据分布情况应当如下表：  

|节点|Block|
|----|----|
|0|(0,0)|
|1|(1,1)|
|2|(2,2)|
|3|(3,3)|

那么每个Block对应的节点应当如下表：  

|Block|节点|
|----|----|
|0|0|
|1|1|
|2|2|
|3|3|

　　block_2_node 属性定义了从block到node的映射，可以是数组，也可以是 dict。该映射关系就表述了表1.中的映射
情况，使用 Hash 或 Array，使得我们能够使用 O(1) 的时间访问到对应的映射关系。
　　node_2_block 属性定义了从node到block的映射，可以是数组，也可以是 dict。该映射关系就表述了表2.中的映射
情况，使用 Hash 或 Array，使得我们能够使用 O(1) 的时间访问到对应的映射关系。
　　blocks 属性定义了所有可用的 block 的编号，返回为数组。
　　block_count 属性返回上述 blocks 属性返回的 block 个数。

## 其他
　　本项目给出的数据划分方案是基于 batch 的，通过给定的长度，将每个 batch 划分为小的 block 并发送给每个计算
节点，重写 Optimizer 或 job_submit 来改动每个节点获取到的样本形式。