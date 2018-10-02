import numpy as np
import pandas as pd
from sklearn.metrics.regression import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from mlens.visualization import corrmat


def rmse_scorer(y_true, y_pred, **kwargs):
    return np.sqrt(mean_squared_error(y_true, y_pred, **kwargs))


def difference_loss(y_true, y_pred):
    return np.subtract(y_true, y_pred)


RMSE_SCORER = make_scorer(rmse_scorer, greater_is_better=False)


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
