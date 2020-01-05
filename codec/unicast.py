from codec.plain import PlainCommunicationCtrl


class UnicastCommunicationCtrl(PlainCommunicationCtrl):

    def update_blocks(self, blockweight):
        # simulate unicast
        for target_set, pack in super().update_blocks(blockweight):
            for target in target_set:
                yield (target, pack)


if __name__ == '__main__':
    ctrl = UnicastCommunicationCtrl(node_id=1)