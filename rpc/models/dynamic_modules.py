import sys
import importlib.util

from typing import TypeVar, Generic, Type

from rpc.models.interface import IReplyPackage


Dynamic_Module_Name = "dynamic_modules"
_spec = importlib.util.spec_from_loader(Dynamic_Module_Name, loader=None)
_module = importlib.util.module_from_spec(_spec)
sys.modules[Dynamic_Module_Name] = _module

T = TypeVar("T")


class ClassSerializer(IReplyPackage, Generic[T]):

    def __init__(self, cls_name: Type[T]):
        # check module
        execution_dirs = cls_name.__module__.split('.')
        # if its from __main__
        if execution_dirs[0] == '__main__':
            import sys
            mod_name = sys.argv[0]
        else:
            # get module filename
            mod_name = "./" + "/".join(execution_dirs) + ".py"

        self.__mod_name = mod_name
        self.__class_name = cls_name.__name__
        self.__mod_content = ''
        with open(mod_name, 'r', encoding='utf-8') as file:
            line = file.readline()
            while line != '' and line[:26] != 'if __name__ == \'__main__\':':
                self.__mod_content += line
                line = file.readline()
        self.__class_type: [Type[T]] = None

    def restore(self) -> Type[T]:
        definitions = compile(self.__mod_content, self.__mod_name, "exec", optimize=2)
        exec(definitions, _module.__dict__)
        cls_type = getattr(_module, self.__class_name)
        self.__class_type = cls_type
        return cls_type

    def __call__(self, *params, **kwargs) -> T:
        if self.__class_type:
            return self.__class_type(*params, **kwargs)
        else:
            return None
