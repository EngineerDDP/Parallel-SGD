# 导入节点分配容器
from network import NodeAssignment, ICommunication_Controller
# 导入请求处理
from network import Request
# 导入自定义执行器基类
from executor.abstract import AbsSimpleExecutor
# 导入网络通信模型
from models import ReplyPackage


class WordCount(AbsSimpleExecutor):

    def __init__(self, node_id: int, working_group: set):
        super().__init__(node_id, working_group)
        self.__data = None

    def satisfy(self, reply: list) -> list:
        for rep in reply:
            self.__data = rep.content()
        return [None]

    def requests(self) -> list:
        return [None]

    def ready(self) -> bool:
        return self.__data is not None

    def run(self, com: ICommunication_Controller) -> None:
        cnt = dict()
        for line in self.__data:
            cnt[line] = cnt.get(line, 0) + 1
        with open("result.txt", 'w') as file:
            file.write(str(len(cnt)))

    def trace_files(self) -> list:
        return ["result.txt"]


if __name__ == '__main__':

    # 添加一个ip作为Worker
    nodes = NodeAssignment()
    nodes.add(101, '127.0.0.1')
    net = Request()

    # 增加协调者角色
    from roles import Coordinator

    # 加载数据集
    with open("wordcnt.txt", 'r') as file:
        lines = file.readlines()

    # 发起一个请求
    # 如果客户端已经启动了，则可以直接提交，无需将代码更新至客户端。
    with net.request(nodes) as req:
        # 在请求的集群上创建一个协调者
        master = Coordinator(req)
        # 提交任务
        master.submit_group(WordCount, package_size=18000)
        # 分发数据
        master.resources_dispatch(lambda x, y: ReplyPackage(lines))
        # 等待执行完成
        master.join()
