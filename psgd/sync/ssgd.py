import time
from queue import Queue, Empty
from typing import List, Set, Iterable, Union

from numpy import ndarray

from codec.essential import BlockWeight
from codec.interfaces import Codec, netEncapsulation, T
from psgd.sync.interface import IParallelSGD
from psgd.sync.interface import ReadTimeOut, AsyncDetected
from utils.constants import SSGD_Sync_Timeout_Limit_MSec


def iterator_helper(objs: Union[Iterable[netEncapsulation[T]], netEncapsulation[T], None]) -> Iterable[netEncapsulation[T]]:
    """
        Retrieve data from generator
    :param objs:
    :return:
    """
    if type(objs).__name__ == 'generator':
        return [i for i in objs]

    if objs is None:
        return []

    return [objs]


class SynchronizedSGD(IParallelSGD):
    """
        For further detail, please check class description of IParallel.
    """

    STR_BATCH_NO = 'SSGD_BATCH_NO'
    DATA = 'SSGD_SUB_DATA'
    INT_READ_TIMEOUT_MSEC = SSGD_Sync_Timeout_Limit_MSec

    def __init__(self, batch_updater: Codec):
        """
            Initialization.
        :param batch_updater: codec class.
        """
        self.__batch_updater: Codec = batch_updater
        self.__current_batch: int = 0
        self.__company_list: List[Set[int]] = []
        self.__adversary_list: List[Set[int]] = []
        self.__receive_buffer: Queue[netEncapsulation[T]] = Queue(maxsize=256)

    @property
    def batch_updater(self):
        return self.__batch_updater

    def release_memory(self):
        """
            release out-dated memory for local batch buffer and codec buffer.
        """
        # remove outdated buffer
        while not self.__receive_buffer.empty():
            self.__receive_buffer.get()

    def update_weights(self, content: ndarray, batch_no: int, block_id: int) -> Iterable[dict]:
        """
            Update weights to the cluster.
            note: only one working process on each node.
                  there can be different working progress among each nodes.
        """
        self.__current_batch = batch_no

        block = BlockWeight(content, block_id)
        update_packs = iterator_helper(self.batch_updater.update_blocks(block))

        for update_pack in update_packs:
            sender = update_pack.target()
            pkg = {
                SynchronizedSGD.STR_BATCH_NO: batch_no,
                SynchronizedSGD.DATA: update_pack
            }
            yield (sender, pkg)

    def accept_data(self, content: dict) -> [Iterable[dict]]:
        """
            Accept object and put it in the queue if the data
            is way ahead of current working progress.
        """
        sender_batch: int = content[SynchronizedSGD.STR_BATCH_NO]
        if self.__current_batch <= sender_batch <= self.__current_batch + 1:
            self.__receive_buffer.put(content[SynchronizedSGD.DATA])
        else:
            raise AsyncDetected()

    def require_weights(self, batch_no: int) -> ndarray:
        """
            Synchronized weights combine.
            Decode all the data after required.
        """
        if self.__current_batch != batch_no:
            raise AsyncDetected()

        time_out_end = time.time() + SynchronizedSGD.INT_READ_TIMEOUT_MSEC / 1000

        while not self.batch_updater.is_done():
            # wait until more data is available
            try:
                pkg: netEncapsulation[T] = self.__receive_buffer.get(timeout=1)
                self.batch_updater.receive_blocks(pkg.content)
            except Empty:
                pass

            if time.time() >= time_out_end:
                # read time out after INT_READ_TIMEOUT_MS million seconds
                raise ReadTimeOut(self.batch_updater.do_something_to_save_yourself)

        self.release_memory()
        return self.batch_updater.get_result()
