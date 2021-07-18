# 导入节点分配容器
# 导入自定义执行器基类
from executor.abstract import AbsSimpleExecutor
from executor.communication import Communication
# 导入网络通信模型
from models import ReplyPackage
from network import NodeAssignment
# 导入请求处理
from network import Request


class WordCount(AbsSimpleExecutor):

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
        return ["Words"]

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

    def run(self, com: Communication) -> object:
        """
            具体任务的执行流程。
        :param com: 任务开始执行时，ICommunication_Controller 的控制权会被转交，所有
                    来自 Coordinator 的信息会被屏蔽，直到该流程退出。
        :return: 返回任意类型，如果该节点不需要报告结果，返回None。
                    该返回值可以使用 Coordinator 中的 join 方法接收。
        """
        # 建字典
        cnt = dict()
        # 逐行检查
        for line in self.__data:
            # 去重
            cnt[line] = cnt.get(line, 0) + 1
        # 返回不重复的单词数目
        return len(cnt)


if __name__ == '__main__':

    # 添加一个ip作为Worker
    nodes = NodeAssignment()
    # 分配 Worker 的 ID 和 IP 地址
    # ID 为整型，不可重复
    nodes.add(101, '127.0.0.1')
    # 请求类
    net = Request()

    # 增加协调者角色
    from roles import Coordinator

    # 加载数据集
    with open("wordcnt.txt", 'r') as file:
        lines = file.readlines()

    # 配置数据集分发逻辑
    def dispatch(node_id: int, request: object) -> ReplyPackage:
        """
            Dispatch 函数，本函数为最简单的实现，暂不关注Worker ID，为每个 Worker 分配
            一样的数据，进而执行一样的操作。
        :param node_id: 忽略掉了 node_id 参数。该参数为节点 id 信息，为 int 型。
        :param request: 请求的类型，即 Executor 中 requests 的返回值。
                        注意：这里 request 不是一个列表，是逐条确认的。
        :return: 返回 IReplyPackage 类型，将有 Coordinator 回复给 Worker 并确认。
        """
        # 如果需要 "Words" 则返回
        if request == "Words":
            # 返回值需要匹配 IReplyPackage 接口，可以自定义类型。在自定义类型时，
            # 定义放在 .py 文件的其他声明的之前，实现 IReplyPackage 接口。
            # 不需要自定义类型时，可以使用默认的 ReplyPackage 类，该类实现了
            # IReplyPackage 接口。
            return ReplyPackage(lines)
        # 如果请求其他数据，没有
        else:
            return ReplyPackage(None)

    # 发起一个请求
    # 如果客户端已经启动了，则可以直接提交，无需将代码更新至客户端。
    with net.request(nodes) as req:
        # 在请求的集群上创建一个协调者
        master = Coordinator(req)
        # 提交任务
        master.submit_group(WordCount, package_size=18000)
        # 分发数据，dispatch_map 参数接收一个 Callable Object，
        # 满足参数类型为 (int, object) 返回值为 (IReplyPackage)
        # resource_dispatch 部分为强类型约束的，需要对类型进行声明。
        # 关于 Python3 的类型约束
        # 参考 pep-0484 : https://www.python.org/dev/peps/pep-0484/
        master.resources_dispatch(dispatch_map=dispatch)
        # 等待执行完成
        cnt = master.join()
        # 返回每个Worker上的最终执行结果，Worker不区分主从，全部按照id排序。
        # 以dict()形式组织：Key为Worker id，Value为返回值。
        print(cnt)
