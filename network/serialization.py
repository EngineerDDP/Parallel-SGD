import numpy as np

from tempfile import TemporaryFile


class Serialize:

    @staticmethod
    def pack(dic:dict):
        with TemporaryFile() as file:
            np.save(file, dic)
            file.seek(0, 0)
            data = file.read()
            # compressed = zlib.compress(data, level=3)
        return data

    @staticmethod
    def unpack(data=b''):
        with TemporaryFile() as file:
            # data = zlib.decompress(data)
            file.write(data)
            while file.tell() != 0:
                file.seek(0, 0)
            pack = np.load(file, allow_pickle=True)[()]
        return pack


class TLVPack:
    Block_Size = 1024 * 1024
    TLV_Type_Normal = 1
    TLV_Type_Exit = 0

    def __init__(self, content:bytes):
        self.Content = content
        self.Length = len(content)

    def send(self, io):
        tlv_package = TLVPack.TLV_Type_Normal.to_bytes(1, 'big') + self.Length.to_bytes(4, 'big') + self.Content
        put = 0
        while put < len(tlv_package):
            put += io.send(tlv_package[put:])

    @staticmethod
    def recv(io):
        type_ = io.recv(1)
        type_ = int.from_bytes(type_, 'big')

        if type_ == TLVPack.TLV_Type_Exit:
            raise OSError('Connection closed by remote computer.')

        length = b''
        while len(length) < 4:
            length += io.recv(4 - len(length))

        length = int.from_bytes(length, 'big')

        content = b''
        take = 0
        while take < length:
            read_len = min(length - take, TLVPack.Block_Size)
            content += io.recv(read_len)
            take = len(content)

        return TLVPack(content)


def request_close(io):
    io.send(TLVPack.TLV_Type_Exit.to_bytes(1, 'big'))

def unpack(io):
    return Serialize.unpack(TLVPack.recv(io).Content)

def pack(data, io):
    TLVPack(Serialize.pack(data)).send(io)