from utils.constants import VERSION


class Version:

    def __init__(self, node_id: int):
        self.__version = VERSION
        self.__node_id = node_id

    def __repr__(self):
        return "Worker: ({}) in Version: ({}).".format(self.__node_id, self.__version)

    def __str__(self):
        return self.__repr__()
