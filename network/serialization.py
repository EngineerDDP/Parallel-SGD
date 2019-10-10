import numpy as np
import zlib

from tempfile import TemporaryFile


class Serialize:

    def pack(dic={}):
        with TemporaryFile() as file:
            np.save(file, dic)
            file.seek(0,0)
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

