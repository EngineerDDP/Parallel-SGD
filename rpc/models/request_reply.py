from rpc.models.interface import IRequestPackage, IReplyPackage


class RequestPackage(IRequestPackage):

    def __init__(self, content: object):
        self.__content = content

    def content(self) -> object:
        return self.__content

    def __repr__(self):
        return "<Requests ({})>".format(self.__content)

    def __str__(self):
        return self.__repr__()


class ReplyPackage(IReplyPackage):

    def __init__(self, content: object):
        self.__content = content

    def restore(self) -> None:
        pass

    def content(self) -> object:
        return self.__content

    def __repr__(self):
        return "<Replys ({})>".format(self.__content)

    def __str__(self):
        return self.__repr__()
