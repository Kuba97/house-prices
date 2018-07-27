import numpy as np


def shuffle_data(x_train, y_train):
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    return x_train, y_train


def mean_squared_error(desired, predicted):
    return np.square(desired - predicted).mean()


def root_mean_squared_error(desired, predicted):
    return np.sqrt(mean_squared_error(desired, predicted))


def single_lambda_mse(model, x_train, y_train, reg_coef, k, fold_size):
    mse_errs = []
    for fold in range(k):
        start_slice = fold * fold_size
        stop_slice = (start_slice + fold_size) if fold < k - 1 else len(x_train)
        train_interval = list(range(start_slice)) + list(range(-len(x_train), -stop_slice))
        val_interval = slice(start_slice, stop_slice)
        model.fit(x_train[train_interval], y_train[train_interval], reg_coef)
        mse_errs.append(root_mean_squared_error(y_train[val_interval], model.predict(x_train[val_interval])))
    return np.mean(mse_errs)


def k_fold_cv(model, x_train, y_train, reg_lambdas, k):
    fold_size = len(x_train) // k
    cv_mean_errs = []
    for reg_coef in reg_lambdas:
        cv_mean_errs.append(single_lambda_mse(model, x_train, y_train, reg_coef, k, fold_size))
    return cv_mean_errs
