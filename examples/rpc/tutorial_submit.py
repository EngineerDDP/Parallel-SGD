# 导入并行调度模块
import random

import rpc
# 导入Object Stream模块
import network


class Sum(rpc.AbsSimpleExecutor):

    def __init__(self, node_id: int, working_group: set, initializer_id):
        """
            构造函数接收的参数是固定的，无法增加新的参数，所需的其他数据要依赖 requests
            方法在构造后请求。
        :param node_id: 指示当前 Executor 在哪个 Worker 上
        :param working_group: 指示当前 Executor 从属于哪个工作组，集合内包含了本组
                            所有节点的id。
        :param initializer_id: 指示当前发起任务的 Coordinator 的ID
        """
        super().__init__(node_id, working_group, initializer_id)
        # 预留 data 变量
        self.__data = None

    def requests(self) -> list:
        """
            Worker 的 Request 列表，返回Worker所需的数据类别，
            Coordinator通过Worker ID和请求类别决定给该Worker返回
            什么数据。
        :return: List[object] 类型。用于标志 Worker 需要什么数据。
        """
        return ["Numbers"]

    def satisfy(self, reply: list) -> list:
        """
            处理从 Coordinator 返回的 Replies ，在网速良好且 Worker 数目不多的
            情况下， Replies 是批量打包返回的。
        :param reply: List[IReplyPackage] 类型，为 Coordinator 依照 Request 请求
                逐条确认返回的数据。数据确保完好送达或未送达，不会送达损坏的数据。
        :return: List[object] 类型，函数返回尚未从此次 Replies 获取到的数据，用于
                通知 Coordinator 本 Worker 尚未收到哪些数据。
        """
        # 使用 for 循环逐个处理
        for rep in reply:
            # 因为 word count 只有一种类别的数据，直接赋值
            self.__data = rep.content()
        # 没有所需的额外数据了。
        return []

    def ready(self) -> bool:
        """
            向 Worker 确认是否可以开始执行。
            每次批量接收 replies 后都会调用该方法确认 Executor 是否可以正常执行了。
        :return: Bool型
        """
        return self.__data is not None

    def run(self, com: rpc.Communication) -> object:
        """
            具体任务的执行流程。
        :param com: rpc 提供的内核调度机制，可以控制进度和切换调度线程，或者处理
                    来自Coordinator的调度信号，终止执行流程。
        :return: 返回任意类型，如果该节点不需要报告结果，返回None。
                    该返回值可以使用 Coordinator 中的 join 方法接收。
        """
        # 建字典
        sum = 0
        # 逐行检查
        for number in self.__data:
            sum += number
        # 返回不重复的单词数目
        return sum


if __name__ == '__main__':
    # 添加一个ip作为Worker
    nodes = network.NodeAssignment()
    # 分配 Worker 的 ID 和 IP 地址
    # ID 为整型，不可重复，不可占用内部ID
    # 这里添加了一个ID为0的Worker，地址为127.0.0.1
    nodes.add(0, '127.0.0.1')
    # 可以添加更多的Worker
    # 请求类
    net = network.Request()

    # 生成一千个随机数
    rnd = random.Random()
    numbers = [rnd.random() for i in range(1000)]
    # 获取当前已经添加的Node数目
    node_cnt = len(nodes)
    # 算一下每个节点能分到多少数据
    numbers_per_node = len(numbers) // node_cnt

    # 配置数据集分发逻辑
    # dispatch_map 参数接收一个 Callable Object，
    # 满足参数类型为 (int, object) 返回值为 (IReplyPackage)
    # resource_dispatch 部分为强类型约束的，需要对类型进行声明。
    # 关于 Python3 的类型约束
    # 参考 pep-0484 : https://www.python.org/dev/peps/pep-0484/
    def dispatch(node_id: int, request: object) -> rpc.ReplyPackage:
        """
            Dispatch 函数
        :param node_id: 忽略掉了 node_id 参数。该参数为节点 id 信息，为 int 型。
        :param request: 请求的类型，即 Sum 类中 requests 的返回值。
                        注意：这里 request 不是一个列表，是逐条确认的。
        :return: 返回 IReplyPackage 类型，将由 Coordinator 回复给 Worker 并确认。
                # 可以自定义返回值类型，返回值需要实现 IReplyPackage 接口。
                # 这里我们没有用自定义类型，而是使用默认的 ReplyPackage 来包装我们的数据，
                # 使用自定义类型可以实现一些预操作，但是意义不大。
        """
        if request == "Numbers":
            numbers_for_this_node = numbers[node_id * numbers_per_node: (node_id + 1) * numbers_per_node]
            return rpc.ReplyPackage(numbers_for_this_node)
        else:
            return rpc.ReplyPackage(None)

    # 发起一个请求
    # 如果客户端已经启动了，则可以直接提交，无需将代码更新至客户端。
    with net.request(nodes) as req:
        # 在请求的集群上创建一个协调者
        master = rpc.Coordinator(req)
        # 提交任务
        master.submit_group(Sum, package_size=18000)
        # 注册数据分发函数
        master.resources_dispatch(dispatch_map=dispatch)
        # 等待执行完成
        # 返回值为两个，第一个元素是执行结果，第二个元素代表执行过程中是否报错
        # 返回每个Worker上的最终执行结果，Worker不区分主从，全部按照id排序。
        # 以dict()形式组织：Key为Worker id，Value为返回值。
        res, err = master.join()

        if not err:
            # 汇总结果
            reduce = 0
            for node_id in res:
                reduce += res[node_id]

            print("We have the result:\t{}.".format(reduce))
        else:
            print("Ops, there was an error during execution.")
