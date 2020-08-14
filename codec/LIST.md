# 内置编码器列表

## Worker 控制器

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

## Server 控制器

|索引字符串|引用路径|简介|
|----|----|----|
|"simple"|codec.naive_ps.ParaServerCodec|参数平均化方法|
|"grad"|codec.naive_ps.GradDiffParaServerCodec|梯度增量更新策略|
|"dc"|codec.dc_asgdcodec.DCASGDServerCodec|带延迟补偿的梯度增量更新策略|
|"sgq"|codec.sgq.SGQServer|基于哈夫曼编码和随机三值化的增量更新策略|
|"qpa"|codec.quantization.FPWParaServer|全精度量化梯度平均化处理策略|