import numpy as np

from tempfile import TemporaryFile
from socket import socket


class BufferReader:
    """
        Buffer reader reads data from io channel.
        Contents were formatted in LV type.
        {length(4 bytes), Value(length bytes)}
        BufferReader raises an OSError when resolving zero length content.
    """
    def __init__(self):
        self.__tmp_files = TemporaryFile()

        self.__content = b''
        self.__length = 0

    def __len__(self):
        return self.__length

    def __clear(self):
        self.__length = 0
        self.__content = b''

    def close(self):
        self.__clear()
        self.__tmp_files.close()

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
            # len(head) == 0 or head == G'0000'
            if self.__length == 0:
                raise OSError('Connection is deprecated.')
        # try read what's left
        if self.__length > len(self.__content):
            self.__content += io.recv(self.__length - len(self.__content))

    def is_done(self) -> bool:
        return self.__length > 0 and (self.__length == len(self.__content))

    def get_content(self) -> dict:
        """
            Get content and clear buffer
        :return: bytes
        """
        res = self.unpack(self.__content)
        self.__clear()
        return res

    def __del__(self):
        self.close()


class BufferWriter:
    """
        Buffer writer writes data to io channel.
        Contents were formatted in LV type.
        {length(4 bytes), Value(length bytes)}
        Can use static method request_close(io) to raise a OSError for io receiver.
    """
    def __init__(self):
        self.__tmp_files = TemporaryFile()

        self.__length = 0
        self.__content = b''

    def __len__(self):
        return self.__length

    def set_content(self, content:[dict, bytes]):
        if self.__length != 0:
            raise Warning('Set content on a buffer which has already had a content.')
        if content is not None:
            if isinstance(content, dict):
                self.__content = self.pack(content)
            elif isinstance(content, bytes):
                self.__content = content
        else:
            raise TypeError('Buffer writer requires something to send.')
        self.__length = len(self.__content)

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

    def close(self):
        self.__tmp_files.close()

    def send(self, io: socket):
        """
            Try write to fd until all the data were sent.
        :param io:
        :return:
        """
        if self.__length == len(self.__content):
            tlv_package = self.__length.to_bytes(4, 'big') + self.__content
            # try send once and record how much bytes has been sent.
            self.__length = self.__length - io.send(tlv_package) + 4
        else:
            self.__length = self.__length - io.send(self.__content[-self.__length:])

    def is_done(self):
        return self.__length == 0

    def __del__(self):
        self.close()
