from abc import abstractmethod, ABCMeta

import numpy as np


class NpModel(metaclass=ABCMeta):
    def __init__(self):
        print(f"create instance for 'class {self.__class__.__name__}'")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class NpLinearRegression(NpModel):
    def __init__(self):
        super().__init__()

    def predict(self):
        pass

    def fit(self):
        pass