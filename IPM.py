from abc import abstractmethod, ABCMeta

class IPM(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
         pass

    @abstractmethod
    def load_data(self, filename):
        return

    @abstractmethod
    def sketch(self, data):
        return

    @abstractmethod
    def fit(self, data):
        return

    @abstractmethod
    def data_preprocess(self):
        return