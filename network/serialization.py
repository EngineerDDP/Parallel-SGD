import numpy as np

from tempfile import TemporaryFile
from socket import socket, MSG_WAITALL


class Buffer:

    Block_Size = 1024*256

    def __init__(self, content:[dict, bytes, None]=None):
        self.__tmp_files = TemporaryFile()

        if content is not None:
            if isinstance(content, dict):
                self.__content = self.pack(content)
            elif isinstance(content, bytes):
                self.__content = content
            else:
                raise TypeError('Buffer content must be dict or bytes.')
            self.__length = len(self.__content)
        else:
            self.__content = b''
            self.__length = 0

    def set_content(self, content:[dict, bytes]):
        if self.__length != 0:
            raise Warning('Set content on a buffer which has already had a content.')
        if content is not None:
            if isinstance(content, dict):
                self.__content = self.pack(content)
            elif isinstance(content, bytes):
                self.__content = content
        self.__length = len(self.__content)

    def __clear(self):
        self.__length = 0
        self.__content = b''

    def close(self):
        self.__clear()
        self.__tmp_files.close()

    @staticmethod
    def request_close(io: socket):
        """
            Send zeros to raise deprecated error and close the connection.
        :param io: socket
        :return: None
        """
        zero_len_mark = int(0).to_bytes(4, 'big')
        io.send(zero_len_mark)

    def pack(self, dic: dict):
        file = self.__tmp_files
        # make sure to write from beginning
        file.truncate(0)
        file.seek(0)
        np.save(file, dic)

        file.seek(0)
        data = file.read()
        # compressed = zlib.compress(data, level=3)
        return data

    def unpack(self, data=b''):
        file = self.__tmp_files
        # data = zlib.decompress(data)
        # write from beginning every time
        file.truncate(0)
        file.seek(0)
        file.write(data)

        file.seek(0)
        pack = np.load(file, allow_pickle=True)[()]

        return pack

    def send(self, io: socket):
        """
            Try write to fd until all the data were sent.
        :param io:
        :return:
        """
        tlv_package = self.__length.to_bytes(4, 'big') + self.__content
        # wither successful or not
        self.__clear()

        put = 0
        io.setblocking(True)
        while put < len(tlv_package):
            end = min(put + Buffer.Block_Size, len(tlv_package))
            put += io.send(tlv_package[put:end])
        io.setblocking(False)

    def recv(self, io: socket):
        """
            Receive once from the fd
        :return:
        """
        # try get header
        if self.__length == 0:
            # assumption that 4 bytes can read at least
            head = io.recv(4)
            self.__length = int.from_bytes(head, 'big')
            # len(head) == 0 or head == b'0000'
            if self.__length == 0:
                raise OSError('Connection is deprecated.')
        # try read what's left
        if self.__length > len(self.__content):
            self.__content += io.recv(self.__length - len(self.__content))

    def is_ready(self) -> bool:
        return self.__length > 0 and (self.__length == len(self.__content))

    def get_content(self) -> dict:
        """
            Get content and clear buffer
        :return: bytes
        """
        res = self.unpack(self.__content)
        self.__clear()
        return res

if __name__ == '__main__':
    from time import time

    begin = time()
    with TemporaryFile() as tmp:
        for i in range(1000):
            tmp.truncate()
            tmp.write(b'ABCDE'*100)
            tmp.seek(0)
    end = time()
    print('scenario 1：',format(end-begin))

    begin = time()
    for i in range(1000):
        with TemporaryFile() as tmp:
            tmp.write(b'ABCDE'*100)
            tmp.seek(0)
    end = time()
    print('scenario 2：', format(end - begin))