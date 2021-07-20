from typing import Hashable, SupportsFloat, Dict

from parallel_sgd.profiles.blockassignment.interface import ISetting
from log import IPrinter


class GlobalSettings:
    deprecated_default_settings: ISetting = None
    deprecated_global_logger: IPrinter = None
    global_parameters: Dict[Hashable, SupportsFloat] = None

    @staticmethod
    def get_default() -> ISetting:
        return GlobalSettings.deprecated_default_settings

    @staticmethod
    def global_logger() -> IPrinter:
        return GlobalSettings.deprecated_global_logger

    @staticmethod
    def get_params(key: Hashable) -> float:
        if GlobalSettings.global_parameters is not None:
            return GlobalSettings.global_parameters.get(key, 0)
        else:
            return 0
