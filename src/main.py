from features.prepare_data import preprocess_procedure
from models.ridge_regression import Ridge
from models.validators import k_fold
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Ridge


def model_selection():
    x, y, ids = preprocess_procedure()
    lambdas = [.001, .01, .1, .6, 1, 10, 100, 1000]
    best_lambda = np.NaN
    means = {}
    for reg_lambda in lambdas:
        errors = []
        kth_fold = k_fold(len(x), 5)
        model = Ridge(reg_lambda)
        for fold_idx in kth_fold:
            train_mask = np.ones(len(x), bool)
            train_mask[fold_idx] = False
            model.fit(x[train_mask], y[train_mask])
            predictions = model.predict(x[~train_mask])
            errors.append(np.sqrt(mse(y[~train_mask], predictions)))
        means[reg_lambda] = np.mean(errors)
    best_lambda = min(means, key=means.get)
    print(best_lambda, ': ', means[best_lambda])


if __name__ == '__main__':
    model_selection()
