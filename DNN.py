from abc import ABC, abstractmethod


class DNN(ABC):

    @abstractmethod
    def _forward_pass(self):
        pass

    @abstractmethod
    def _get_data(self):
        pass

    @abstractmethod
    def _build_network(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
