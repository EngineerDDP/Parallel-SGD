import datetime


class Logger:

    def __init__(self, title_info, log_to_file=False):

        self.Title = title_info
        self.ToFile = log_to_file
        self.File_Name = 'log_file_{}_{}.log'.format(title_info, datetime.datetime.now().strftime('%H-%M-%S'))

    def log_message(self, msg):

        time = datetime.datetime.now()
        time_str = time.strftime('%H:%M:%S')
        str = 'INFO {}@{} : {}'.format(self.Title, time_str, msg)

        print(str)
        self.__log_to_file(str)

    def log_error(self, error):

        time = datetime.datetime.now()
        time_str = time.strftime('%H:%M:%S')
        str = 'ERROR @ {} : {}'.format(time_str, error)

        print(str)
        self.__log_to_file(str)

    def __log_to_file(self, msg):

        if self.ToFile:
            with open(self.File_Name, 'a+') as file:
                file.write(msg + '\n')