from codec import GlobalSettings
from codec.essential import BlockWeight
from codec.interfaces import Codec
from codec.interfaces import netEncapsulation


class MyComCtrl(Codec):
    def __init__(self, node_id):
        super().__init__(node_id)
        self.__global_weights = 0
        self.__current_recv = 0

    def dispose(self):
        # 使用 record 方法记录信息到 P-SGD 的 log 文件中
        self.record('my communication controller is disposed.')

    def update_blocks(self, block_weight: BlockWeight):
        self.record('Weights delta received.')
        self.record('from block: {}'.format(block_weight.block_id))
        self.record('It has a content with shape: {}'.format(block_weight.content.shape))

        # 获取没有该数据的节点
        send_to = block_weight.adversary
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
        return netEncapsulation(send_to, pkg)

    def receive_blocks(self, content: dict):
        self.record('I have received an package.')
        self.record('It has a content with shape: {}'.format(content['data'].shape))
        # 记录梯度内容
        self.__global_weights += content['data']
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
