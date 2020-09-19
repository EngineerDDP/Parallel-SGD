from executor.abstract import AbsSimpleExecutor
from network.interfaces import ICommunication_Controller

# 继承自 AbsSimpleExecutor
class myExecutor(AbsSimpleExecutor):

    def __init__(self, node_id, offset):
        """
            Node id 是分配到的全局node_id，offset是本组起始id的值
            在当前抽象层面上，Executor可以访问集群内任意节点，每个节点都有独立id，
            分组offset仅仅是分组建议，并非强制不可见。

            在 Codec 层面上，分组策略才是强制执行的，每个组内有重复的id，组与组之间
            不可互访。
        """
        super().__init__(node_id, offset)
        # 计算结果以文件的形式记录
        self.__my_output_file = "./Done({})".format(self.node_id)

    def requests(self) -> list:
        return []

    def satisfy(self, reply:list) -> list:
        return []

    def start(self, com: ICommunication_Controller) -> None:
        """
            获取 ICommunication_Controller 控制权，开始执行任务。
        """
        with open(self.__my_output_file, 'w+') as f:
            for i in range(1, 21):
                f.write("{} ".format(i))
        import shutil as sh
        try:
            sh.rmtree('./tmp_log')
        except:
            pass

    def trace_files(self) -> list:
        """
            返回当前执行结果保存在哪些文件上。
        """
        return [self.__my_output_file]

