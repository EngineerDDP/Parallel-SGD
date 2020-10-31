import datetime
import os
from abc import ABCMeta, abstractmethod


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

    def log_message(self, msg):

        time = datetime.datetime.now()
        time_str = time.strftime('%H:%M:%S')
        str = 'INFO {}@{} : {}'.format(self.Title, time_str, msg)

        print(str)
        self.__log_to_file(str)

    def log_error(self, error):

        time = datetime.datetime.now()
        time_str = time.strftime('%H:%M:%S')
        str = 'ERROR {}@{} : {}'.format(self.Title, time_str, error)

        print(str)
        self.__log_to_file(str)

    def __log_to_file(self, msg):
        if not os.path.exists(self.Folder):
            os.mkdir(self.Folder)

        if self.ToFile:
            with open(self.File_Name, 'a+') as file:
                file.write(msg + '\n')


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
