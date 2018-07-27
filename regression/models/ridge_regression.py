import numpy as np
from models.linear_model import LinearModel


class Ridge(LinearModel):
    def __init__(self, coef_shape):
        super(Ridge, self).__init__(coef_shape)

    def fit(self, x_train, y_train, reg_lambda=0):
        x_transposed = x_train.T
        reg_lambda *= np.identity(x_train.shape[1])
        self.coef = np.linalg.inv(x_transposed.dot(x_train) - reg_lambda).dot(x_transposed).dot(y_train)
