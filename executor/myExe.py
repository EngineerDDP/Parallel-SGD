from dataset.interfaces import IDataset
from executor.interfaces import AbsSimpleExecutor
from models.local import IServerModel
from network.interfaces import ICommunication_Controller
from profiles import Settings


class myExecutor(AbsSimpleExecutor):

    def __init__(self, node_id, offset):
        super().__init__(node_id, offset)
        self.__my_output_file = "./output.txt"

    def run(self, com: ICommunication_Controller) -> None:
        with open(self.__my_output_file, 'a') as f:
            f.write('Hello world!')

    def trace_files(self) -> list:
        return [self.__my_output_file]
