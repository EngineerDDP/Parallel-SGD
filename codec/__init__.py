from profiles import ISetting
from utils.log import IPrinter


class GlobalSettings:
    deprecated_default_settings: ISetting = None
    deprecated_global_logger: IPrinter = None

    @staticmethod
    def get_default() -> ISetting:
        return GlobalSettings.deprecated_default_settings

    @staticmethod
    def global_logger() -> IPrinter:
        return GlobalSettings.deprecated_global_logger
