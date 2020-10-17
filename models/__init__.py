import os
from abc import ABCMeta, abstractmethod
from typing import Set


class IRequestPackage(metaclass=ABCMeta):

    @abstractmethod
    def content(self) -> object:
        pass


class IReplyPackage(metaclass=ABCMeta):

    @abstractmethod
    def restore(self) -> None:
        pass


class RequestWorkingLog:

    def __init__(self):
        pass


class ReadyType:

    def __init__(self, nodes_ready: set):
        self.__nodes_ready = nodes_ready

    def current_ready(self):
        return self.__nodes_ready


class DoneType(IReplyPackage):

    def __init__(self, node_id, posted_files):
        self.__header = "./Node-{}-Retrieve/".format(node_id)
        self.__contents = [BinaryFilePackage(f) for f in posted_files]

    def restore(self) -> None:
        for bf in self.__contents:
            bf.filename = self.__header + bf.filename
            bf.restore()

    @property
    def file_list(self):
        for bf in self.__contents:
            yield bf.filename


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


class ClassSerializer(IReplyPackage):

    def __init__(self, cls_name: type):
        # check module
        execution_dirs = cls_name.__module__.split('.')
        # if its from __main__
        if execution_dirs[0] == '__main__':
            import sys
            mod_name = sys.argv[0]
        else:
            # get module filename
            mod_name = "./" + "/".join(execution_dirs) + ".py"

        self.__class_name = cls_name.__name__
        self.__mod_content = ''
        with open(mod_name, 'r', encoding='utf-8') as file:
            line = file.readline()
            while line != '' and line[:26] != 'if __name__ == \'__main__\':':
                self.__mod_content += line
                line = file.readline()
        self.__class_type: [type] = None

    def restore(self) -> type:
        import importlib
        spec = importlib.util.spec_from_loader('codec_module', loader=None)
        mod = importlib.util.module_from_spec(spec)

        exec(self.__mod_content, mod.__dict__)
        cls_type = getattr(mod, self.__class_name)
        self.__class_type = cls_type
        return cls_type

    def __call__(self, *params, **kwargs):
        if self.__class_type:
            return self.__class_type(*params, **kwargs)
        else:
            return None


class SubmitJob(IReplyPackage):

    def __init__(self, nodes: set, eta_waiting_time: int, exe: type):
        self.__nodes = nodes
        self.__eta_wait = eta_waiting_time
        self.__cls: [ClassSerializer, type] = ClassSerializer(exe)

    def restore(self) -> None:
        self.__cls = self.__cls.restore()

    @property
    def executioner(self) -> [type]:
        return self.__cls

    @property
    def work_group(self) -> Set[int]:
        return self.__nodes

    @property
    def waiting_time(self) -> [float, int]:
        return self.__eta_wait
