from psgd.sync.ssgd import SynchronizedSGD, iterator_helper


class AsynchronizedSGD(SynchronizedSGD):
    """
        For further detail, please check class description of IParallel.
        Implemented in accordance with HogWild:
        *Absence*
    """

    INT_BATCH_SKIP = -1

    def __init__(self, codec):
        super().__init__(codec)

    def release_memory(self):
        """
            Currently not used.
        :return: None
        """
        pass

    def accept_data(self, obj):
        """
            Accept the data and get weights updated immediately.
        :param obj: json like object: encode
        :return: None
        """
        # check if the data was outdated
        sender_batch = obj[SynchronizedSGD.STR_BATCH_NO]
        if sender_batch > AsynchronizedSGD.INT_BATCH_SKIP:
            # get package iterable
            pack_to_sends = iterator_helper(self.batch_updater.receive_blocks(obj[SynchronizedSGD.DATA].content()))

            # iterate package
            for pack_to_send in pack_to_sends:
                target = pack_to_send.target()
                # tag this layer
                pkg = {
                    SynchronizedSGD.STR_BATCH_NO: sender_batch,
                    SynchronizedSGD.DATA: pack_to_send
                }
                yield target, pkg

    def require_weights(self, tag):
        return self.batch_updater.get_result()
