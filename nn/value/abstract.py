__var_id_max = 0
__var_ids = set()


def register():
    global __var_id_max, __var_ids
    __var_ids.add(__var_id_max)
    __var_id_max += 1
    return __var_id_max - 1


class AbsValue:

    def __init__(self):
        self.var_id = register()

    @property
    def id(self):
        return self.var_id

    def __repr__(self):
        return "<Value ID:{}>".format(self.id)
