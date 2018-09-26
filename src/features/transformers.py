import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class FunctionTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, func, **kwargs):
        self.func = func
        self.func_kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X, **self.func_kwargs)

    def get_params(self, deep=True):
        return {'func': self.func, **self.func_kwargs}


class FeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, feats_to_drop=None):
        if feats_to_drop is None:
            feats_to_drop = []
        self.feats_to_drop = feats_to_drop

    def fit(self, X, y=None):
        self.feats_to_drop = [feat for feat in self.feats_to_drop if feat in X.columns.tolist()]
        return self

    def transform(self, y):
        return y.drop(self.feats_to_drop, axis=1)

    def get_params(self, deep=True):
        return {'feats_to_drop': self.feats_to_drop}


class TypeSelector(TransformerMixin, BaseEstimator):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, y):
        return y.select_dtypes(include=self.dtype).copy()

    def get_params(self, deep=True):
        return {'dtype': self.dtype}


class Imputer(TransformerMixin, BaseEstimator):
    CONST_METHOD = 'const'
    MEAN_METHOD = 'mean'
    MEDIAN_METHOD = 'median'
    MODE_METHOD = 'mode'

    def __init__(self, features, method='const', value=None):
        self.features = features
        self.method = method
        self.impute_values = [value] * len(features)
        self.simple_value = value

    def fit(self, X, y=None):
        self.features = [feat for feat in self.features if feat in X.columns.tolist()]
        self.impute_values = (self._compute_fill_values(X)).iloc[0]
        return self

    def transform(self, y):
        return y.fillna(self.impute_values)

    def get_params(self, deep=True):
        return {'features': self.features, 'method': self.method, 'value': self.simple_value}

    def _compute_fill_values(self, X):
        if self.method == Imputer.CONST_METHOD:
            impute_values = pd.DataFrame([self.simple_value] * len(self.features), index=self.features).transpose()
        elif self.method == Imputer.MEAN_METHOD:
            impute_values = X[self.features].mean(axis=0)
        elif self.method == Imputer.MEDIAN_METHOD:
            impute_values = X[self.features].median(axis=1)
        elif self.method == Imputer.MODE_METHOD:
            impute_values = X[self.features].mode(axis=0)
        else:
            raise ValueError(self.method + ' not in available imputing methods')
        return impute_values


class DummyEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, drop_first=False):
        self.drop_first = drop_first

    def fit_transform(self, X, y=None, **fit_params):
        X = pd.get_dummies(X, drop_first=self.drop_first)
        self.dummied_features = X.columns
        return X

    def fit(self, X, y=None):
        self.dummied_features = pd.get_dummies(X, drop_first=self.drop_first).columns
        return self

    def transform(self, y, *_):
        y = pd.get_dummies(y, drop_first=self.drop_first)
        return y.reindex(columns=self.dummied_features, fill_value=0)

    def get_params(self, deep=True):
        return {'drop_first': self.drop_first}


class FeatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, func, feats_to_transform=None):
        if feats_to_transform is None:
            feats_to_transform = []
        self.feats_to_transform = feats_to_transform
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, y):
        y = self.func(y, self.feats_to_transform)
        return y

    def get_params(self, deep=True):
        return {'feats_to_transform': self.feats_to_transform, 'func': self.func}


class FeatureBucketizer(TransformerMixin, BaseEstimator):
    def __init__(self, feature_name, q_num):
        self.feature_name = feature_name
        self.q_num = q_num

    def fit(self, X, y=None):
        _, self.bins = pd.qcut(X[self.feature_name], self.q_num, retbins=True)
        return self

    def transform(self, y):
        y[self.feature_name] = pd.cut(y[self.feature_name], self.bins)
        return y

    def fit_transform(self, X, y=None, **fit_params):
        X[self.feature_name], self.bins = pd.qcut(X[self.feature_name], self.q_num, retbins=True)
        return X


class LogTransform(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, y):
        return np.log(y)

    def inverse_transform(self, y):
        return np.exp(y)
