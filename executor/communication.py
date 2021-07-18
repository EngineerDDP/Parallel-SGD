from typing import Tuple

import executor.models as models
import network


class Communication:

    def __init__(self, com: network.ICommunicationController, initializer_id: int):
        self.__com = com
        self.__initializer_id = initializer_id
        self.__interruptions: bool = False
        # Redirect functions
        self.send_one = com.send_one

    def block_send(self, target, value, timeout):
        raise InterruptedError("This job has been aborted.")

    def get_one(self, blocking, timeout) -> Tuple[int, object]:
        id_from, content = self.__com.get_one(blocking, timeout)
        if isinstance(content, models.Kill):
            self.__interruptions = True
            self.send_one = self.block_send
            raise InterruptedError("This job has been aborted.")
        else:
            return id_from, content

    # Redirect properties
    @property
    def available_clients(self):
        return self.__com.available_clients

    @property
    def available_clients_count(self):
        return self.__com.available_clients_count

    # Redirect properties

    # Add property access
    @property
    def bytes_sent(self):
        return self.__com.Com.bytes_sent

    @property
    def bytes_read(self):
        return self.__com.Com.bytes_read

    # Add property access

    def report_progress(self, progress: int):
        assert progress <= 100 and progress >= 0, "Progress value must between 0 and 100"
        self.__com.send_one(self.__initializer_id, models.Progress(progress))
