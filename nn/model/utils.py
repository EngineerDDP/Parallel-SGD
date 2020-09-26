class FitResultHelper:

    def __init__(self):
        self.__history_result = []
        self.__title = []

    @property
    def history(self):
        return self.__history_result

    @property
    def title(self):
        return self.__title

    @property
    def count(self):
        return len(self.__history_result)

    def set_fit_title(self, title):
        self.__title = title

    def append_row(self, row: list):
        eval_str = ', '.join(["{}:{:.4f}".format(key, val) if isinstance(val, float) else "{}:{}".format(key, val)
                              for key, val in zip(self.__title, row)])
        self.__history_result.append(row)
        return eval_str
