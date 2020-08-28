from codec.interfaces import ICommunication_Ctrl
from codec.essential import Block_Weight
from codec.interfaces import netEncapsulation

from codec import GlobalSettings
import numpy as np

class toy_codec(ICommunication_Ctrl):
    def __init__(self, node_id):
        super().__init__()
        # 保存并记录本节点编号信息，除此之外再也没有其他地方可以获取该信息
        self.__node_id = node_id
        # 存储收集的所有梯度之和
        self.__global_weights = 0
        self.__current_recv = 0

        # 保留本节点从初始化到现在累计更新的的梯度之和
        self.__weight = 0

    def dispose(self):
        print('my communication controller is disposed.')

    # class Block_Weight:
    #     """
    #         Weights calculated using one block
    #     """
    #
    #     def __init__(self, layer_id, batch_id, block_id, company_id, content):
    #         self.Layer_ID = layer_id 本参数的层编号
    #         self.Batch_ID = batch_id 本参数的批次编号
    #         self.Block_ID = block_id 本参数的样本Block编号
    #         self.Company_ID = company_id 本参数对应的样本Block还在哪些节点上出现过
    #         # calculate who doesnt have these block
    #         self.Adversary_ID = set(range(GlobalSettings.get_default().node_count)) - set(company_id) 本参数对应的样本Block没在哪些节点上出现过
    #         self.Content = content 	参数本身

    def update_blocks(self, block_weight: Block_Weight):

        # 获取没有该数据的节点
        send_to = block_weight.Adversary_ID
        # 我们使用 'data' 字符串来标记我们的梯度内容
        self.__weight -= block_weight.Content
        content = np.mean(self.__weight)
        pkg = {
            'data': content
            ,"shape": block_weight.Content.shape
        }
        # 记录本机梯度
        self.__global_weights += self.__weight
        self.__current_recv += 1
        # 检查是否接受完所有数据
        self.__do_grad_average()
        # 发送梯度
        yield netEncapsulation(send_to, pkg)

    def receive_blocks(self, content: dict):

        # 记录梯度内容
        mean_data = content['data']
        data_shape = content['shape']

        self.__global_weights += np.random.normal(size=data_shape,loc=mean_data)

        # 记录已经接收到多少个梯度了
        self.__current_recv += 1
        # 检查是否接受完所有数据
        self.__do_grad_average()

    def __do_grad_average(self):
        how_much_nodes = GlobalSettings.get_default().node_count
        if self.__current_recv == how_much_nodes:
            # 执行梯度平均
            self.set_result(self.__global_weights / how_much_nodes)
            # 重设梯度值，等待下一批次的循环
            self.__global_weights = 0
            self.__current_recv = 0

    def do_something_to_save_yourself(self):
        print("放弃治疗")
