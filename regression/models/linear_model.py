from abc import ABC, abstractmethod
import numpy as np


class LinearModel(ABC):
    def __init__(self, coef_shape):
        self.coef = np.empty(coef_shape)

    @abstractmethod
    def fit(self, x_train, y_train, reg_lambda=0):
        pass

    def predict(self, x):
        return x.dot(self.coef)
