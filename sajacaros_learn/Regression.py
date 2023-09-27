from abc import abstractmethod, ABCMeta

import numpy as np


class NpModel(metaclass=ABCMeta):
    def __init__(self):
        print(f"create instance for 'class {self.__class__.__name__}'")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, lr: float) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class NpLinearRegression(NpModel):
    def __init__(self):
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass