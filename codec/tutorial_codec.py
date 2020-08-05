from codec.interfaces import ICommunication_Ctrl
from codec.essential import Block_Weight
from codec.interfaces import netEncapsulation

from profiles.settings import GlobalSettings


class myComCtrl(ICommunication_Ctrl):
    def __init__(self, node_id):
        super().__init__()
        # 保存并记录本节点编号信息，除此之外再也没有其他地方可以获取该信息
        self.__node_id = node_id
        self.__global_weights = 0
        self.__current_recv = 0

    def dispose(self):
        print('my communication controller is disposed.')

    def update_blocks(self, block_weight: Block_Weight):
        print('Weights delta received.')
        print('from block: {}'.format(block_weight.Block_ID))
        print('It has a content with shape: {}'.format(block_weight.Content.shape))

        # 获取没有该数据的节点
        send_to = block_weight.Adversary_ID
        # 我们使用 'data' 字符串来标记我们的梯度内容
        pkg = {
            'data': block_weight.Content
        }
        # 记录本机梯度
        self.__global_weights += block_weight.Content
        self.__current_recv += 1
        # 检查是否接受完所有数据
        self.__do_grad_average()
        # 发送梯度
        yield netEncapsulation(send_to, pkg)

    def receive_blocks(self, json_dict: dict):
        print('I have received an package.')
        print('It has a content with shape: {}'.format(json_dict['data'].shape))
        # 记录梯度内容
        self.__global_weights += json_dict['data']
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