import matplotlib.pyplot as plt
import pandas as pd

from data import Data
from models.ridge_regression import Ridge
from utils import k_fold_cv
from submission_utils import save_submission

LABEL_NAME = 'SalePrice'

def plot_error(x, y):
    plt.plot(x, y)
    plt.xlabel('lambda')
    plt.ylabel('mse(log)')
    plt.ylim([0, 1])
    plt.show()


if __name__ == '__main__':
    train = Data(label_col_name='SalePrice', id_col_name='Id')
    train.add_intercept()
    train.to_matrix()

    reg_coef = [.001, .01, .05, .1, .5, 1, 2, 5, 10]
    current_model = Ridge(train.x_train.shape[1]+1)
    errors = k_fold_cv(current_model, train.x_train, train.y_train, reg_coef, 5)
    print(errors)
    # save_submission(pd.Series(errors), "ridge", min(errors))
    plot_error(reg_coef, errors)
