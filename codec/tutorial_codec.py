from codec import GlobalSettings
from codec.interfaces import Codec
from codec.essential import BlockWeight
from codec.interfaces import netEncapsulation


class MyComCtrl(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        # 保存并记录当前批次已经收到了多少份结果
        self.__global_weights = 0
        self.__current_recv = 0

    def dispose(self):
        print('my communication controller is disposed.')

    def update_blocks(self, block_weight: BlockWeight):
        print('Weights delta received.')
        print('from block: {}'.format(block_weight.block_id))
        print('It has a content with shape: {}'.format(block_weight.content.shape))

        # 获取没有该数据的节点
        send_to = GlobalSettings.get_default().get_adversary(block_weight.block_id)
        # 我们使用 'data' 字符串来标记我们的梯度内容
        pkg = {
            'data': block_weight.content
        }
        # 记录本机梯度
        self.__global_weights += block_weight.content
        self.__current_recv += 1
        # 检查是否接受完所有数据
        self.__do_grad_average()
        # 发送梯度
        yield netEncapsulation(send_to, pkg)

    def receive_blocks(self, json_dict: dict):
        print('I have received an package.')
        print('It has a content with shape: {}'.format(json_dict['data'].shape))
        # 我们使用上述定义的 'data' 字符串获取我们更新的梯度内容
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