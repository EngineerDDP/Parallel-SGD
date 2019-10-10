class Delegation(object):

    def __init__(self):

        self.__Listener_Methods = []

    def __add__(self, value):

        if isinstance(value, Delegation):
            self.__Listener_Methods += value.__Listener_Methods
        if callable(value):
            self.__Listener_Methods.append(value)

        return self

    def clear(self):

        self.__Listener_Methods.clear()

    def invoke(self, *params):

        for listener in self.__Listener_Methods:
            # start_new_thread(listener, params)
            listener(*params)

        return


