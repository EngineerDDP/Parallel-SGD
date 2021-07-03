import datetime
import os
from abc import ABCMeta, abstractmethod

BUFFERED_MESSAGE_MAX = 1000


class IPrinter(metaclass=ABCMeta):

    @abstractmethod
    def log_message(self, msg):
        pass

    @abstractmethod
    def log_error(self, error):
        pass



class Logger(IPrinter):

    def __init__(self, title_info, log_to_file=False):

        self.Title = title_info
        self.ToFile = log_to_file
        self.Folder = './tmp_log/'
        self.File_Name = self.Folder + '{} {}.log'.format(title_info, datetime.datetime.now().strftime('%Y-%m-%d %H%M'))
        self.__file = self.__open_or_create()

        self.__buffer = ''
        self.__buffered_msg_count = 0
        self.__buffered_msg_max = BUFFERED_MESSAGE_MAX

    def log_message(self, msg):

        time = datetime.datetime.now()
        time_str = time.strftime('%H:%M:%S')
        str_highlighted = "\033[0;32;mINFO\033[0m \033[1;36;m{}\033[0;32;m@\033[1;32;m{}\033[0m: {}".format(self.Title, time_str, msg)
        str_save = "INFO {}@{}: {}".format(self.Title, time_str, msg)

        print(str_highlighted)
        self.__log_to_file(str_save)

    def log_error(self, error):

        time = datetime.datetime.now()
        time_str = time.strftime('%H:%M:%S')
        str_highlighted = "\033[1;31;mERROR\033[0m \033[1;36;m{}\033[0;32;m@\033[1;32;m{}\033[0m: {}".format(self.Title, time_str, error)
        str_save = "ERROR {}@{}: {}".format(self.Title, time_str, error)

        print(str_highlighted)
        self.__log_to_file(str_save)

    def flush(self):
        # flush all
        self.__buffered_msg_max = 1
        self.__log_to_file('')
        self.__buffered_msg_max = BUFFERED_MESSAGE_MAX

    def close(self):
        self.flush()
        if self.__file is not None:
            self.__file.close()

    def __del__(self):
        self.close()

    def __open_or_create(self):
        if not os.path.exists(self.Folder):
            os.mkdir(self.Folder)
        return open(self.File_Name, "a+")

    def __log_to_file(self, msg):
        # Use buffer
        if self.__buffered_msg_count < self.__buffered_msg_max:
            self.__buffered_msg_count += 1
            self.__buffer += msg + '\n'
            return

        if self.ToFile:
            self.__file.write(self.__buffer)
            self.__file.flush()

        self.__buffer = ''
        self.__buffered_msg_count = 0

    # 2021-07-02 修正:
    # 为了保障基于多进程的集成测试能够正常完成。
    def __getstate__(self):
        return self.Title, self.ToFile

    def __setstate__(self, state):
        self.__init__(*state)
    # -------------------------------------


class MutePrinter(IPrinter):

    def __init__(self, title_info, log_to_file=False):
        self.Title = title_info
        self.ToFile = log_to_file
        self.Folder = './tmp_log/'
        self.File_Name = self.Folder + '{} {}.log'.format(title_info, datetime.datetime.now().strftime('%Y-%m-%d %H%M'))

    def log_message(self, msg):
        time = datetime.datetime.now()
        time_str = time.strftime('%H:%M:%S')
        str = 'INFO {}@{} : {}'.format(self.Title, time_str, msg)
        self.__log_to_file(str)

    def log_error(self, error):
        time = datetime.datetime.now()
        time_str = time.strftime('%H:%M:%S')
        str = 'ERROR {}@{} : {}'.format(self.Title, time_str, error)
        self.__log_to_file(str)

    def __log_to_file(self, msg):
        if not os.path.exists(self.Folder):
            os.mkdir(self.Folder)

        if self.ToFile:
            with open(self.File_Name, 'a+') as file:
                file.write(msg + '\n')
