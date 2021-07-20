import pickle
from socket import socket
from tempfile import TemporaryFile

LV_HEADER_LENGTH = 4


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
        self.__obj: object = None
        self.__length = LV_HEADER_LENGTH
        self.__done_mark = True

    def __len__(self):
        return self.__length

    def __clear(self):
        self.__length = LV_HEADER_LENGTH
        self.__content = b''
        self.__obj = None
        self.__done_mark = True

    def close(self):
        self.__clear()
        self.__tmp_files.close()

    def deserialize_obj(self, data: bytes):
        file = self.__tmp_files
        # data = zlib.decompress(data)
        # write from beginning every time
        file.truncate(0)
        file.seek(0)
        file.write(data)

        file.seek(0)
        pack = pickle.load(file)
        return pack

    def unpack(self, data: bytes):
        if self.__done_mark:
            self.__length = int.from_bytes(data, "big")
            self.__done_mark = False
        else:
            self.__obj = self.deserialize_obj(data)
            self.__done_mark = True

    def recv(self, io: socket):
        """
            Receive once from the fd
        :return:
        """
        self.__content += io.recv(self.__length - len(self.__content))
        # len(head) == 0 or head == G'0000'
        if len(self.__content) == 0:
            raise OSError('Connection is deprecated.')
        elif len(self.__content) == self.__length:
            self.unpack(self.__content)
            self.__content = b''

    def is_done(self) -> bool:
        return self.__obj is not None

    def get_content(self) -> object:
        """
            Get content and clear buffer
        :return: bytes
        """
        obj = self.__obj
        self.__clear()
        return obj

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

    def set_content(self, content: object):
        if self.__length != 0:
            raise Warning('Set content on a buffer which has already had a content.')
        if content is not None:
            self.__content = self.pack(content)
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

    def serialize_obj(self, obj: object):
        file = self.__tmp_files
        # make sure to write from beginning
        file.truncate(0)
        file.seek(0)
        pickle.dump(obj, file)

        file.seek(0)
        data = file.read()
        # compressed = zlib.compress(data, level=3)
        return data

    def pack(self, obj: object):
        data = self.serialize_obj(obj)
        # Add LV specified header
        data = len(data).to_bytes(4, "big") + data
        return data

    def close(self):
        self.__tmp_files.close()

    def send(self, io: socket):
        """
            Try write to fd until all the data were sent.
        :param io:
        :return:
        """
        self.__length = self.__length - io.send(self.__content[-self.__length:])

    def is_done(self):
        return self.__length == 0

    def __del__(self):
        self.close()

# 2021-06-25更新：
# 1. 在设置发送缓冲区的时候就使用长度填充前四个字节，防止网速过低的时候发送失败。
# 2. 取消对byte数组的额外处理，修正编码解码的不匹配。
