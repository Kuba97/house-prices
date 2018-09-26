import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV

from models.model_selection.utils import RMSE_SCORER


CV_FOLDS = 5


def grid_search_cv(model, x_train, y_train, param_grid, cv_folds=CV_FOLDS):
    gs_model = GridSearchCV(model, param_grid, scoring=RMSE_SCORER, cv=cv_folds, n_jobs=4, verbose=0)
    gs_model.fit(x_train, y_train)
    print('Best score: ', gs_model.best_score_)
    print('Best params: \n', gs_model.best_params_)
    return gs_model.best_estimator_, gs_model.best_score_


def cv_score(model, x_train, y_train, cv_folds=CV_FOLDS):
    scores = cross_val_score(model, x_train, y_train, scoring=RMSE_SCORER, cv=cv_folds, n_jobs=1)
    print_cv_results(scores)


def print_cv_results(scores):
    mean_score = np.mean(scores)
    print(f'Folds scores: {np.negative(scores)}')
    print(f'Mean score: {abs(mean_score)}')
