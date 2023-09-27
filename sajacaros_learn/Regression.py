from abc import abstractmethod, ABCMeta

import numpy as np


def mean_squared_error(y, y_hat):
    return np.mean(np.square(y - y_hat))


class NpModel(metaclass=ABCMeta):
    def __init__(self):
        print(f"create instance for 'class {self.__class__.__name__}'")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, lr: float, it: int) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class NpLinearRegression(NpModel):
    def __init__(self):
        super().__init__()
        self._coef = None
        self._lr = None

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, it: int = 1000) -> None:
        self._lr = lr
        X_b = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        self._coef = np.random.rand(X_b.shape[1])  # features + b

        print(X_b.shape, y.shape, self._coef)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
