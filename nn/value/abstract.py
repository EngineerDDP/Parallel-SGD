from nn.interface import IValue

__var_id_max = 0
__var_ids = set()

def register():
    global __var_id_max, __var_ids
    __var_ids.add(__var_id_max)
    __var_id_max += 1
    return __var_id_max - 1

def remove(id):
    pass

class AbsValue:

    def __init__(self):
        self.__var_id = register()

    def __del__(self):
        remove(self.__var_id)

    @property
    def id(self):
        return self.__var_id
