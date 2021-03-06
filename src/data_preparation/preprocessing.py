from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

import data_preparation.func_transformers as f_trans
import data_preparation.transformers as trans
from data_preparation.features_info import NUMERIC_FEATS, CATEGORIC_FEATS

OUTLIERS_INDICES = [30, 495, 533, 691, 916, 968, 1182]
NUMBER_TYPE_NAME = 'number'
OBJECT_TYPENAME = 'object'
FEATS_TO_DROP = ['PoolQC', 'TotalBsmtSF', 'GarageArea', 'MiscVal', 'MoSold', 'YrSold', 'GarageYrBlt', 'LowQualFinSF',
                 'BsmtFinSF2', '3SsnPorch', 'BsmtHalfBath', 'Condition2', 'GarageCond', 'BsmtFinType2']
FEATS_TO_LOG = ['1stFlrSF', 'GrLivArea', 'LotArea', 'MasVnrArea', 'OpenPorchSF']
FEATS_TO_CAT = ['GarageCars', 'Fireplaces', 'MSSubClass', 'BedroomAbvGr']
VAL_IMPUTE_NUM = 0
VAL_IMPUTE_CAT = 'n/a'
MODE_IMPUTE_CAT_FEATS = ['Street', 'LandContour', 'Utilities', 'LandSlope', 'Condition1', 'Condition2', 'BldgType',
                         'RoofMatl', 'ExterCond', 'Heating', 'CentralAir', 'Electrical', 'Alley', 'Functional',
                         'SaleCondtition']
MEDIAN_IMPUTE_NUM_FEATS = ['LotFrontage', 'OverallQual', 'OverallCond']
ANTIQUE_BOUND = 1930

BASIC_PREPROCESSOR_STEPS = [
    ('encode_nominal', trans.DummyEncoder()),
]

PREPROCESSOR_STEPS = [
    ('is_antique', trans.FunctionTransformer(f_trans.is_antique, antique_bound=ANTIQUE_BOUND)),
    ('discretizer', trans.FunctionTransformer(f_trans.to_categorical, feats_to_cat=FEATS_TO_CAT)),
    ('feat_union', FeatureUnion(n_jobs=1, transformer_list=[
        ('numeric', Pipeline(steps=[
            ('selector', trans.TypeSelector(NUMBER_TYPE_NAME)),
            ('basic_imputer', trans.Imputer(NUMERIC_FEATS, trans.Imputer.CONST_METHOD, VAL_IMPUTE_NUM)),
            ('log_transform', trans.FunctionTransformer(f_trans.log_transform, feats_to_log=FEATS_TO_LOG)),
            ('scaler', StandardScaler())
        ])),
        ('categorical', Pipeline(steps=[
            ('selector', trans.TypeSelector(OBJECT_TYPENAME)),
            ('mode_imputer', trans.Imputer(MODE_IMPUTE_CAT_FEATS, trans.Imputer.MODE_METHOD, VAL_IMPUTE_CAT)),
            ('basic_imputer', trans.Imputer(CATEGORIC_FEATS, trans.Imputer.CONST_METHOD, VAL_IMPUTE_CAT)),
            ('encode_nominal', trans.DummyEncoder())
        ]))
    ])),
]

BASIC_PREPROCESSOR = Pipeline(steps=BASIC_PREPROCESSOR_STEPS)
PREPROCESSOR = Pipeline(steps=PREPROCESSOR_STEPS)


def delete_outliers(x, y, outliers_idx=OUTLIERS_INDICES):
    return x.drop(outliers_idx, axis=0), y.drop(outliers_idx, axis=0)


def make_pipeline_with_estimator(estimator, preprocessor=BASIC_PREPROCESSOR):
    model = preprocessor
    model.steps.append([type(estimator).__name__, estimator])
    return model
