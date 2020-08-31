from executor.interfaces import AbsSimpleExecutor
from network.interfaces import ICommunication_Controller

# 继承自 AbsSimpleExecutor
class myExecutor(AbsSimpleExecutor):

    def __init__(self, node_id, offset):
        """
            Node id 是分配到的全局node_id，offset是本组起始id的值
        """
        super().__init__(node_id, offset)
        # 计算结果以文件的形式记录
        self.__my_output_file = "./HelloWorld.txt"

    def run(self, com: ICommunication_Controller) -> None:
        """
            获取 ICommunication_Controller 控制权，开始执行任务。
        """
        with open(self.__my_output_file, 'w+') as f:
            for i in range(1, 21):
                f.write("{} ".format(i))

    def trace_files(self) -> list:
        """
            以文件的形式返回执行结果
        """
        return [self.__my_output_file]

