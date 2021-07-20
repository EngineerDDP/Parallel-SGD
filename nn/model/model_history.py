import time


class FitResultHelper:

    def __init__(self):
        self.__history_result = []
        self.__title = []
        self.__start_time = time.time()

    @property
    def history(self):
        return self.__history_result

    @property
    def title(self):
        return ["Time"] + self.__title

    @property
    def count(self):
        return len(self.__history_result)

    def set_fit_title(self, title):
        self.__title = title

    def append_row(self, row: list):
        eval_str = ', '.join(["{}:{:.4f}".format(key, val) if isinstance(val, float) else "{}:{}".format(key, val)
                              for key, val in zip(self.__title, row)])
        row.insert(0, float(time.time() - self.__start_time))
        self.__history_result.append(row)
        return eval_str

    def __getstate__(self):
        return self.__title

    def __setstate__(self, state):
        self.__title = state
        self.__history_result = []
        self.__start_time = time.time()
