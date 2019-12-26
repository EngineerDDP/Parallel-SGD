import datetime


class Logger:

    def __init__(self, title_info, mute=True):

        self.Title = title_info
        self.Mute = mute

    def log_message(self, msg):

        time = datetime.datetime.now()
        time_str = time.strftime('%H:%M:%S')

        if not self.Mute:
            print('INFO {}@{} : {}'.format(self.Title, time_str, msg))

    def log_error(self, error):

        time = datetime.datetime.now()
        time_str = time.strftime('%H:%M:%S')

        if not self.Mute:
            print('ERROR @ {} : {}'.format(time_str, error))