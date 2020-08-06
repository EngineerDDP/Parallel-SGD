import numpy as np

from tempfile import TemporaryFile
from socket import socket


def pack(dic:dict):
    with TemporaryFile() as file:
        np.save(file, dic)
        file.seek(0, 0)
        data = file.read()
        # compressed = zlib.compress(data, level=3)
    return data

def unpack(data=b''):
    with TemporaryFile() as file:
        # data = zlib.decompress(data)
        file.write(data)
        while file.tell() != 0:
            file.seek(0, 0)
        pack = np.load(file, allow_pickle=True)[()]
    return pack


class Buffer:

    def __init__(self, content:[dict, bytes, None]=None):
        if content is not None:
            if isinstance(content, dict):
                self.__content = pack(content)
            elif isinstance(content, bytes):
                self.__content = content
            else:
                raise TypeError('Buffer content must be dict or bytes.')
            self.__length = len(self.__content)
        else:
            self.__content = b''
            self.__length = 0

    @staticmethod
    def request_close(io: socket):
        """
            Send zeros to raise deprecated error and close the connection.
        :param io: socket
        :return: None
        """
        zero_len_mark = int(0).to_bytes(4, 'big')
        io.send(zero_len_mark)

    def send(self, io: socket):
        """
            Try write to fd until all the data were sent.
        :param io:
        :return:
        """
        tlv_package = self.__length.to_bytes(4, 'big') + self.__content

        put = 0
        while put < len(tlv_package):
            put += io.send(tlv_package[put:])

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
        return self.__length != 0 and (self.__length == len(self.__content))

    def get_content(self) -> dict:
        """
            Get content and clear buffer
        :return: bytes
        """
        res = unpack(self.__content)
        self.__content = b''
        self.__length = 0
        return res

