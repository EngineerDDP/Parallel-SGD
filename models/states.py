from typing import List

from models.binary_file import BinaryFilePackage
from models.interface import IReplyPackage


class ReadyType:

    def __init__(self, nodes_ready: set):
        self.__nodes_ready = nodes_ready

    def current_ready(self):
        return self.__nodes_ready


class DoneType(IReplyPackage):

    def __init__(self, node_id: int, posted_files: List[str], returns: object = None, exps: [Exception] = None):
        self.__node_id = node_id
        self.__header = "./Node-{}-Retrieve/".format(node_id)
        self.__contents = [BinaryFilePackage(f) for f in posted_files]
        self.__returns = returns
        self.__exceptions = exps

    def restore(self) -> None:
        for bf in self.__contents:
            bf.filename = self.__header + bf.filename
            bf.restore()

    @property
    def result(self):
        return self.__returns

    @property
    def exception(self):
        return self.__exceptions

    @property
    def file_list(self):
        for bf in self.__contents:
            yield bf.filename

    def __str__(self):
        return "<Node({}) Reply: All Task is Completed.>".format(self.__node_id)

    def __repr__(self):
        return "<DoneType from Node({})>".format(self.__node_id)
