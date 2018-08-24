import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import Ridge, RidgeCV, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

from features.transformers import FeatureSelector, CustomImputer, OrdinalEncoder, DummyEncoder, FeatureTransformer
from features_info import CATEGORIC_ORDINAL, CATEGORIC_NOMINAL, NUMERIC_FEATS, CATEGORIC_FEATS, TARGET_COL
from math import sqrt

NAME_ESTIMATOR = 'estimator'
CV_FOLDS = 5
RIDGE_ALPHAS = [10 ** x for x in range(-3, 4)]


def make_preprocess_pipeline():
    preprocessed = Pipeline(memory=None,
                            steps=[
                                ('drop_features', FeatureSelector()),
                                ('impute_numeric', CustomImputer()),
                                ('impute_categorical', CustomImputer()),
                                ('encode_ordinal', OrdinalEncoder()),
                                ('encode_nominal', DummyEncoder()),
                                ('log_transform', FeatureTransformer()),
                                ('standard_scale', Normalizer())
                            ])
    return preprocessed


FEATS_TO_DROP = ['LotFrontage', 'GarageYrBlt']
VAL_IMPUTE_NUM = 0
VAL_IMPUTE_CAT = 'None'
ORDINAL_ORDER_STAT = '50%'
DROP_FIRST = True
FEATS_TO_LOG = ['1stFlrSF', 'GrLivArea']


def set_preprocessing_params(preprocessing, **params):
    preprocessing.set_params(
        drop_features__feats_to_drop=FEATS_TO_DROP,
        impute_numeric__features=NUMERIC_FEATS,
        impute_numeric__value=VAL_IMPUTE_NUM,
        impute_categorical__features=CATEGORIC_FEATS,
        impute_categorical__value=VAL_IMPUTE_CAT,
        encode_ordinal__ordinal_feats=CATEGORIC_ORDINAL,
        encode_ordinal__target_to_order=params['target_to_order'],
        encode_ordinal__order_statistic=ORDINAL_ORDER_STAT,
        encode_nominal__drop_first=False,
        log_transform__feats_to_transform=FEATS_TO_LOG,
        log_transform__func=np.log
    )


def append_estimator(pipeline, pipe_name, estimator):
    regression = Pipeline(steps=[(pipe_name, pipeline),
                                 (estimator.__name__, Ridge())
                                 ])
    return regression


def build_model_pipeline(**params):
    preprocessing = make_preprocess_pipeline()
    set_preprocessing_params(preprocessing, **params)
    model_pipeline = append_estimator(preprocessing, 'preprocessing', params['estimator'])
    return model_pipeline


def run_validation(train):
    x_train = train.x
    y_train = np.log(train.y)
    estimator = Ridge
    model_pipeline = build_model_pipeline(target_to_order=y_train, estimator=Ridge)
    param_grid = {
        estimator.__name__+'__alpha': [10**x for x in range(-4, 4)]
    }
    cv_model = GridSearchCV(model_pipeline, param_grid=param_grid, cv=CV_FOLDS, scoring='neg_mean_squared_error')
    cv_model.fit(x_train, y_train)
    print(cv_model.cv_results_)
    print('Best score: ', sqrt(abs(cv_model.best_score_)))
    pass
