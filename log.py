import datetime


class Logger:

    Mute = True

    def __init__(self, title_info):

        self.Title = title_info

    def log_message(self, msg):

        time = datetime.datetime.now()
        time_str = time.strftime('%H:%M:%S')

        if not Logger.Mute:
            print('INFO @ {} : {}'.format(time_str, msg))

    def log_error(self, error):

        time = datetime.datetime.now()
        time_str = time.strftime('%H:%M:%S')

        if not Logger.Mute:
            print('ERROR @ {} : {}'.format(time_str, error))