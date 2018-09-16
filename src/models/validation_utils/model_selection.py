import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

from data_io.make_dataset import Data
import features.func_transformers as f_trans
import features.transformers as trans
from features.features_info import NUMERIC_FEATS, CATEGORIC_FEATS
from .validator import Validator

NUMBER_TYPENAME = 'number'
OBJECT_TYPENAME = 'object'
FEATS_TO_DROP = ['PoolQC', 'TotalBsmtSF', 'GarageArea', 'MoSold', 'YrSold', 'MiscVal', 'GarageYrBlt', 'LowQualFinSF',
                 'BsmtFinSF2', '3SsnPorch', 'LotFrontage', 'BsmtHalfBath', 'Condition2', 'GarageCond', 'BsmtFinType2']
FEATS_TO_LOG = ['1stFlrSF', 'GrLivArea', 'LotArea', 'BsmtFinSF1', 'MasVnrArea', 'OpenPorchSF']
FEATS_TO_CAT = ['GarageCars', 'Fireplaces', 'MSSubClass', 'BedroomAbvGr']
VAL_IMPUTE_NUM = 0
VAL_IMPUTE_CAT = 'n/a'
ANTIQUE_BOUND = 1930


def make_preprocess_pipeline():
    preprocessing = Pipeline(steps=[
        ('drop_features', trans.FeatureSelector(FEATS_TO_DROP)),
        ('is_antique', trans.FunctionTransformer(f_trans.is_antique, antique_bound=ANTIQUE_BOUND)),
        ('discretizer', trans.FunctionTransformer(f_trans.to_categorical, feats_to_cat=FEATS_TO_CAT)),
        ('feat_union', FeatureUnion(n_jobs=1, transformer_list=[
            ('numeric', Pipeline(steps=[
                ('selector', trans.TypeSelector(NUMBER_TYPENAME)),
                ('basic_imputer', trans.SimpleImputer(NUMERIC_FEATS, VAL_IMPUTE_NUM)),
                ('log_transform', trans.FunctionTransformer(f_trans.log_transform, feats_to_log=FEATS_TO_LOG)),
                ('standardize', StandardScaler())
            ])),
            ('categorical', Pipeline(steps=[
                ('selector', trans.TypeSelector(OBJECT_TYPENAME)),
                ('basic_imputer', trans.SimpleImputer(CATEGORIC_FEATS, VAL_IMPUTE_CAT)),
                ('encode_nominal', trans.DummyEncoder())
            ]))
        ])),
    ])
    return preprocessing


def validate(train):
    x_train = train.x
    y_train = np.log(train.y).values
    final_estimator = ElasticNet(alpha=0.001, l1_ratio=.5, max_iter=256)
    validator = Validator(make_preprocess_pipeline(), final_estimator)

    cv_score(x_train, y_train, validator)
    # grid_search_cv(x_train, y_train, validator)
    # train_store_predictions(validator, x_train, y_train)
    # train_store_coefs(validator, x_train, y_train)


def cv_score(x_train, y_train, validator):
    scores = validator.get_cross_val_scores(x_train, y_train)
    print_cv_results(scores)


def grid_search_cv(x_train, y_train, validator):
    param_grid = {}  # fill dictionary with desired parameters to perform grid_search
    gs_model = validator.grid_search(x_train, y_train, param_grid)
    print(log_rmse_score(gs_model.best_score_))
    print(gs_model.best_params_)


def log_rmse_score(log_mse_score):
    return np.sqrt(abs(log_mse_score))


def print_cv_results(scores):
    folds_scores = log_rmse_score(scores)
    mean_score = log_rmse_score(np.mean(scores))
    print(f'Folds scoring: {folds_scores}')
    print(f'Mean scoring: {mean_score}')