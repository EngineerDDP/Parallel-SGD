from queue import Queue
from time import sleep

from codec.essential import Block_Weight
from psgd.interfaces import IParallelSGD
from psgd.interfaces import ReadTimeOut, AsyncDetected, OutdatedUpdates


def iterator_helper(iter):
    """
        Retrieve data from generator
    :param iter:
    :return:
    """
    if type(iter).__name__ == 'generator':
        return [i for i in iter]

    if iter is None:
        return []

    return [iter]


class SynchronizedSGD(IParallelSGD):
    """
        For further detail, please check class description of IParallel.
    """

    STR_BATCH_NO = 'SSGD_BATCH_NO'
    DATA = 'SSGD_SUB_DATA'
    INT_READ_TIMEOUT_MS = 10000

    def __init__(self, node_id, layer_id, codec):
        """
            Initialization.
        :param node_id: the identification of current node.
        :param layer_id: the identification of working layer.
        """

        super().__init__(node_id, layer_id, codec)
        self.receive_buffer = {}
        self.batch_updater = None
        self.current_batch = 0

        self.init_startup_setting()

    def init_startup_setting(self, params=None):
        """
            Currently not used.
        :param params: None
        :return: None
        """
        self.batch_updater = self.Updater(self.Node_ID)

    def release_memory(self):
        """
            release out-dated memory for local batch buffer and codec buffer.
        """
        # remove outdated buffer
        for key in self.receive_buffer.keys():
            if key < self.current_batch - 10:
                del self.receive_buffer[key]

    def update_weights(self, content, tag):
        """
            Update weights to the cluster.
            note: only one working process on each node.
                  there can be different working progress among each nodes.
        """
        self.current_batch = tag.Batch_No

        block = Block_Weight(tag.Layer_No, tag.Batch_No, tag.Block_No, tag.Company, content=content)

        update_packs = iterator_helper(self.batch_updater.update_blocks(block))

        for update_pack in update_packs:
            sender = update_pack.target()
            dic = update_pack.content()
            pkg = {
                SynchronizedSGD.STR_BATCH_NO: tag.Batch_No,
                SynchronizedSGD.DATA: dic
            }
            yield (sender, pkg)

    def accept_data(self, obj):
        """
            Accept object and put it in the queue if the data
            is way ahead of current working progress.
        """
        sender_batch = obj[SynchronizedSGD.STR_BATCH_NO]
        if sender_batch >= self.current_batch:
            self.receive_buffer[sender_batch] = self.receive_buffer.get(sender_batch, Queue())
            self.receive_buffer[sender_batch].put(obj[SynchronizedSGD.DATA])
        else:
            raise OutdatedUpdates()

    def require_weights(self, tag):
        """
            Synchronized weights combine.
            Decode all the data after required.
        """
        if self.current_batch != tag.Batch_No:
            raise AsyncDetected()

        time_out = 0

        while not self.batch_updater.is_done():
            # wait until more data is available
            if self.receive_buffer.get(self.current_batch) is None \
                    or self.receive_buffer[self.current_batch].empty():
                sleep(0.001)
                time_out += 1
                if time_out == SynchronizedSGD.INT_READ_TIMEOUT_MS:
                    # read time out after INT_READ_TIMEOUT_MS million seconds
                    raise ReadTimeOut(self.batch_updater.do_something_to_save_yourself)

            else:
                pkg = self.receive_buffer[self.current_batch].get()
                self.batch_updater.receive_blocks(pkg)

        if self.batch_updater.is_done():
            return self.batch_updater.get_result()
