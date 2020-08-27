from dataset.interfaces import IDataset
from executor.interfaces import IExecutor
from models.local import IServerModel
from network.interfaces import ICommunication_Controller
from profiles import Settings


class myExecutor(IExecutor):

    def __init__(self, node_id):
        super().__init__(node_id)
        self.__my_output_file = "./output.txt"

    def add_info(self, obj: IServerModel):
        pass

    def add_data(self, obj: IDataset):
        pass

    def add_setting(self, obj: Settings):
        pass

    def run(self, com: ICommunication_Controller) -> None:
        with open(self.__my_output_file, 'a') as f:
            f.write('Hello world!')

    def ready(self) -> bool:
        return True

    def done(self) -> bool:
        return True

    def trace_files(self) -> list:
        return [self.__my_output_file]
