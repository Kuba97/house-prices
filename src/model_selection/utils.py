import numpy as np
import pandas as pd
from sklearn.metrics.regression import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from mlens.visualization import corrmat
from utils.io_utils import save_csv, pickle_file, FITTED_MODEL_PATH, REPORT_PATH, REPORT_FORMAT, \
    tag_filename_with_datetime, DUMP_FORMAT


def log_rmse_scorer(y_true, y_pred, **kwargs):
    return np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred), **kwargs))


def difference_loss(y_true, y_pred):
    return np.subtract(y_true, y_pred)


LOG_RMSE_SCORER = make_scorer(log_rmse_scorer, greater_is_better=False)


def check_models_correlation(models, x_train, y_train):
    models_losses = pd.DataFrame()
    for m in models:
        models_losses[m.steps[-1][0]] = cv_loss(m, x_train, y_train)
    corrmat(models_losses.corr(method='pearson'))


def cv_loss(model, x_train, y_train, loss=difference_loss):
    cv_gen = KFold()
    losses = np.empty(shape=[len(x_train)])
    for train_idx, val_idx in cv_gen.split(x_train):
        model.fit(x_train.iloc[train_idx], y_train[train_idx])
        losses[val_idx] = (loss(y_train[val_idx], model.predict(x_train.iloc[val_idx])))
    return losses


def store_cv_results(cv_results, basic_filename, additional_info=''):
    PARAM_NAMES = ['params', 'split0_train_score', 'split1_train_score', 'split2_train_score', 'split3_train_score',
                   'split4_train_score', 'mean_train_score', 'std_train_score']
    cv_results = {i: cv_results[i] for i in cv_results if i not in PARAM_NAMES}
    results_report = pd.DataFrame.from_dict(cv_results)
    filename = tag_filename_with_datetime(basic_filename, REPORT_FORMAT, additional_info)
    save_csv(results_report, filename, REPORT_PATH)


def store_fitted_model(fitted_model, additional_info=''):
    basic_filename = type(fitted_model).__name__
    filename = tag_filename_with_datetime(basic_filename, DUMP_FORMAT, additional_info)
    pickle_file(fitted_model, filename, FITTED_MODEL_PATH)
