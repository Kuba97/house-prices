import time
import numpy as np

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from model_selection.utils import LOG_RMSE_SCORER

from utils.make_dataset import load_split_dataset
from data_preparation.features_info import ID_COL_NAME, TARGET_COL_NAME

import models.estimators as pool
from data_preparation.preprocessing import delete_outliers, make_pipeline_with_estimator, BASIC_PREPROCESSOR
from .utils import store_cv_results

CV_FOLDS = 5


def validate():
    id_, x_train, y_train = prepare_data()
    model = prepare_model()
    cv_score(model, x_train, y_train)
    # hyper_opt(model, x_train, y_train)
    # estimator.fit(x_train, y_train)
    # store_fitted_model(estimator)


def prepare_data():
    id_, x_train, y_train = load_split_dataset('raw', 'train.csv', TARGET_COL_NAME, ID_COL_NAME)
    x_train, y_train = delete_outliers(x_train, y_train)
    return id_, x_train, y_train


def prepare_model():
    estimator = pool.randomized_prep_xgboost
    model = make_pipeline_with_estimator(estimator, BASIC_PREPROCESSOR)
    model = TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)
    return model


def hyper_opt(model, x_train, y_train):
    verbosity = 1
    start_time = time.time()
    hyper_opt_results = randomized_search(model, x_train, y_train, store_validation_info=True, verbosity=verbosity)
    stop_time = time.time()
    show_hyper_opt_info(hyper_opt_results, stop_time - start_time)


def grid_search_cv(model, x_train, y_train, cv_folds=CV_FOLDS, verbosity=0):
    param_grid = {}  # fill grid to seach the space of hyperparameters
    gs_validator = GridSearchCV(model, param_grid, scoring=LOG_RMSE_SCORER, cv=cv_folds, n_jobs=4, verbose=verbosity)
    gs_validator.fit(x_train, y_train)
    return gs_validator


def randomized_search(model, x_train, y_train, verbosity=0, store_validation_info=False):
    param_distributions = {
        'XGBRegressor__max_depth': [2, 3, 4, 5, 6, 7, 8],
        'XGBRegressor__n_estimators': [128, 256, 512, 1024, 2048],
        'XGBRegressor__reg_alpha': [0.1 * x for x in range(1, 20)],
        'XGBRegressor__learning_rate': [0.01, .05, .4, .1, .3, .2, .8, .9, .75, .67, .014, .09, .08, .085, .06],
        'XGBRegressor__colsample_bytree': [0.1 * x for x in range(1, 11)],
        'XGBRegressor__colsample_bylevel': [0.1 * x for x in range(1, 11)],
        'XGBRegressor__subsample': [0.1 * x for x in range(1, 11)],
        'XGBRegressor__reg_lambda': [0.1 * x for x in range(1, 20)],
        'XGBRegressor__gamma': [0, 0.1, .01, .5, .8, .3, .4],
        'XGBRegressor__min_child_weight': [1, 2, 3, 4, 8, 16, 32]
    }
    rs_validator = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=128, n_jobs=4, cv=CV_FOLDS,
                                      scoring=LOG_RMSE_SCORER, verbose=verbosity)
    rs_validator.fit(x_train, y_train)
    if store_validation_info:
        store_cv_results(rs_validator.cv_results_, type(model._final_estimator).__name__, 'randomized_search')
    return rs_validator.best_estimator_


def show_hyper_opt_info(hyper_opt_model, duration):
    print('Best score: ', hyper_opt_model.best_score_)
    print('Best params: \n', hyper_opt_model.best_params_)
    print('Evaluated in: ', str(duration) + 's' if duration < 60 else str(duration / 60) + 'min')


def cv_score(model, x_train, y_train, cv_folds=CV_FOLDS):
    scores = cross_val_score(model, x_train, y_train, scoring=LOG_RMSE_SCORER, cv=cv_folds, n_jobs=1)
    show_cv_score_info(scores)


def show_cv_score_info(scores):
    mean_score = np.mean(scores)
    print(f'Folds scores: {np.negative(scores)}')
    print(f'Mean score: {abs(mean_score)}')
