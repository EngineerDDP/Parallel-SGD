import os

from executor.models.interface import IReplyPackage


class BinaryFilePackage(IReplyPackage):

    def __init__(self, filename):
        self.filename = filename
        self.content = b''
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.content = f.read()

    def restore(self):
        path, file = os.path.split(self.filename)
        if path != '' and not os.path.exists(path):
            os.makedirs(path)

        with open(self.filename, 'wb+') as f:
            f.write(self.content)
