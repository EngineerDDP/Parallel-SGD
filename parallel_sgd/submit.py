from typing import Hashable, SupportsFloat, Type, Union, Dict, Tuple

from network import NodeAssignment, Request

from nn.dataset.transforms.interface import ITransformer
from nn.dataset.interfaces import AbsDataset
from nn.gradient_descent import ADAMOptimizer
from nn.gradient_descent.interface import IGradientDescent
from nn.model import Model
from nn.optimizer.parallel_sgd import PSGDOptimizer, IOptimizer

from parallel_sgd.parameter_server import PSGDPSExecutor
from parallel_sgd.worker import PSGDWorkerExecutor
from parallel_sgd.codec.interfaces import Codec
from parallel_sgd.codec.dummy import DummyCodec
from parallel_sgd.profiles import IIDBlockAssignment
from parallel_sgd.profiles.blockassignment.interface import ISetting
from parallel_sgd.profiles.batch_iter import BatchIter
from parallel_sgd.profiles.blockassignment.abstract import AbsBlockAssignment
from parallel_sgd.batch_sync import AsynchronizedSGD, SynchronizedSGD
from parallel_sgd.batch_sync.sync.interface import ISyncType
from parallel_sgd.net_package import net_model, net_setting, net_transfer, net_optimizer
from parallel_sgd.net_package import misc_package, extra_package, data_package, data_content, Req

from constants import Parameter_Server
from executor import Coordinator
from log import Logger


class ParallelSGD:
    """
        P-SGD 主调类
        P-SGD RPC Controller
    """

    def __init__(self, model: Model, data: AbsDataset, transform: ITransformer):
        """
            初始化一个P-SGD主调对象
        :param model: 用于数据并行化的模型。
        :param data: 用于并行化的数据集。
        :param transform: 数据集处理转换策略。Dataset在每个节点上都是可见的，由 transform 对数据集
                          进行状态转换处理，数据集的处理操作是本地化执行的，数据集转换策略是链表形式
                          组织的，以管道的形式运行，数据集依次经过每个 transform 操作，最终进入由
                          BatchIter 调度。
        """
        self.__model = model
        self.__data = data
        self.__transform = transform
        self.__log = Logger(title_info="P-SGD Submit", log_to_file=True)

    def parallel(self,
                 nodes: NodeAssignment,
                 redundancy: int = 1,
                 block_size: int = 64,
                 epoch: int = 10,
                 assignment_type: Type[AbsBlockAssignment] = IIDBlockAssignment,
                 sync_type: Type[ISyncType] = SynchronizedSGD,
                 op_type: Type[IOptimizer] = PSGDOptimizer,
                 gd_type: Type[IGradientDescent] = ADAMOptimizer,
                 codec: Union[Dict[int, Type[Codec]], Type[Codec]] = None,
                 gd_params: Tuple[object] = (),
                 ps_codec: Union[Dict[int, Type[Codec]], Type[Codec], None] = None,
                 network_bandwidth: int = 1048576,
                 mission_title: str = "P-SGD",
                 ssgd_timeout_limit: int = 10000,
                 codec_extra_parameters: Dict[Hashable, SupportsFloat] = None) -> Dict[str, SupportsFloat]:
        """
            执行并行化。
        :param ssgd_timeout_limit: Sync-SGD等待超时限制，单位为毫秒，数值为整型。
        :param network_bandwidth: 可用的网络带宽，用作计算预估传输时间，设置 pre_commit 超时计时器。
        :param mission_title:   任务标题，作为本次任务的log文件文件名。
        :param nodes:           由 network 模块提供的 NodeAssignment 接口，指示了当前并行化操作调用的节点数目。
                                参数服务器的节点编号由 utils.constant.Parameter_Server 指定，其余工作节点的id
                                从 0 开始依次递增（为整数型）。
        :param redundancy:      冗余设置，适用于能够处理冗余的 codec 和 block assignment。
                                继承自 AbsBlockAssignment 总能够处理含有冗余参数的提交任务，codec 的冗余处理则由
                                codec 自行定义。
        :param block_size:      节点粒度的 Batch 大小，由 codec 控制具体的更新策略，块大小与批次大小并没有具体对应关系。
                                若 codec 在每个 Block 后给出结果，则实践意义上 Block size 和 Batch size 是等价的，
                                若 codec 总是等待所有 Block 的训练完成后再同步结果，则实践意义上 Batch size 等于 Block size
                                乘以 Block 总数。
        :param epoch:           训练批次数，由 codec 和 sync type 共同决定 epoch 内的同步策略，当使用参数服务器时，参数服务器
                                也参与到同步状态的维持中。
                                若 codec 不允许异步执行，则所有节点都会在同一时刻结束 epoch，若 codec 或 sync type 允许跨批次
                                执行，则节点会根据自己的计算能立先后结束计算。
        :param assignment_type: 样本分配策略，一般与冗余分配结合使用，需要实现了 profiles.ISetting 接口的类型。
                                初衷是提供冗余的数据集分配策略，现可以提供静态数据量分配。
        :param sync_type:       同步方式。分同步和异步两种，需要实现了 parallel_sgd.sync.IParallelSGD 接口的类型。
                                同步方式下，每个 worker 在调用 get_weights() 获取权重时才会处理接收数据。
                                异步方式下，每个 Worker 收到数据就处理并更新结果集。
                                具体的数据处理流程和结果集更新策略都由 codec 定义。
        :param gd_type:         梯度处理策略类型，实现了 nn.IOptimizer 接口的类型。
                                负责处理梯度更新策略。
        :param op_type:         梯度生成策略，实现了 nn.gradient_descent.IGradientDescent 接口的类型。
                                负责生成待处理的更新增量。
        :param codec:           编码器类型，实现了 codec.interface.Codec 接口的类型。
        :param gd_params:       梯度生成器参数
        :param ps_codec:        编码器类型，实现了 codec.interface.Codec 接口的类型。
                                用于参数服务器进行数据处理。
        :param codec_extra_parameters:用于Codec接口识别的其他参数列表，为字典形式。该字典将会存储在每个Worker的
                                codec.GlobalSettings.__global_parameters 参数中，Codec对象可以使用
                                GlobalSettings.get_params(key: str) -> object 函数获取对应key的值。
        :return: Dict，代表全局执行结果，平均准确率和损失，以及在Model中定义的其他评判指标。
        """
        # 初始化适合的Codec
        if codec_extra_parameters is None:
            codec_extra_parameters = {}
        if codec is None:
            codec = dict()
        if ps_codec is None:
            ps_codec = dict()

        # 默认填充Codec
        default_codec = DummyCodec
        default_ps_codec = DummyCodec
        # 如果传入确定的Codec
        if isinstance(codec, type):
            default_codec = codec
            codec = dict()
        if isinstance(ps_codec, type):
            default_ps_codec = ps_codec
            ps_codec = dict()

        # 获取所有的合法Slave
        node_count = 0
        has_ps = False
        for _id, _ in nodes:
            if _id >= 0:
                node_count += 1
            else:
                has_ps = True

        # 任务分配策略
        assignment: ISetting = assignment_type(node_count, redundancy)
        # 分配策略实例
        setting: net_setting = net_setting(assignment_type, node_count, redundancy)
        # 模型实例
        model: net_model = net_model(self.__model, BatchIter(block_size, assignment.block_count))
        # 优化器实例
        optimizer: net_optimizer = net_optimizer(op_type, gd_type, op_params=gd_params)
        # 变量表
        var_ids = [var.id for var in self.__model.trainable_variables()]
        # 变量表Codec字典
        var_codec = {var_id: (sync_type, codec.get(var_id, default_codec)) for var_id in var_ids}
        # Transfer 实例
        transfer_worker: net_transfer = net_transfer(var_codec)
        # PS Codec 变量表字典
        var_ps_codec = {var_id: (AsynchronizedSGD, ps_codec.get(var_id, default_ps_codec)) for var_id in var_ids}
        # PS Transfer 实例
        transfer_ps: [net_transfer] = net_transfer(var_ps_codec) if has_ps else None
        # 其他信息
        misc: misc_package = misc_package(mission_title, epoch, None, ssgd_timeout_limit)

        replies = {
            Req.Model: model,
            Req.Setting: setting,
            Req.Optimizer: optimizer,
            Req.Transfer: transfer_worker,
            Req.Transfer_PS: transfer_ps,
            Req.Other_Stuff: misc,
            Req.Extra_Content: extra_package(codec_extra_parameters),
            Req.Data_Package: data_package(self.__data, self.__transform),
            Req.Data_Content: data_content(self.__data, self.__transform)
        }

        req = Request()
        self.__log.log_message("Start job.")
        self.__log.log_message("Workers: {}".format(nodes))

        with req.request(nodes) as com:
            coordinator = Coordinator(com, estimate_bandwidth=network_bandwidth, logger=self.__log)
            if has_ps:
                coordinator.submit_single(PSGDPSExecutor, Parameter_Server, self.__data.estimate_size())
            coordinator.submit_group(PSGDWorkerExecutor, assignment.nodes, self.__data.estimate_size())

            coordinator.resources_dispatch(lambda _id, x: replies[x])
            res, err = coordinator.join()
            self.__log.close()

        # 获取每个节点的返回值
        ret: Dict[str, float] = {}
        for node in res:
            if isinstance(res, Dict) and isinstance(res[node], Dict):
                for key in res[node]:
                    ret[key] = ret.get(key, 0) + res[node][key]

        # 求平均值
        for key in ret:
            ret[key] = ret[key] / len(res)

        return ret
