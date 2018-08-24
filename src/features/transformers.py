import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class FeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, feats_to_drop=None):
        if feats_to_drop is None:
            feats_to_drop = []
        self.feats_to_drop = feats_to_drop

    def fit(self, X, y=None):
        self.feats_to_drop = [feat for feat in self.feats_to_drop if feat in X.columns.tolist()]
        return self

    def transform(self, y, ):
        return y.drop(self.feats_to_drop, axis=1)

    def get_params(self, deep=True):
        return {'feats_to_drop': self.feats_to_drop}


class CustomImputer(TransformerMixin, BaseEstimator):
    def __init__(self, features=None, value=None):
        self.features = features
        self.value = value

    def fit(self, X, y=None):
        self.features = [feat for feat in self.features if feat in X.columns]
        return self

    def transform(self, y):
        # pd.set_option('mode.chained_assignment', 'None')
        if self.features or self.value is not None:
            for feature in self.features:
                y[feature].fillna(self.value, inplace=True)
        return y

    def get_params(self, deep=True):
        return {'features': self.features, 'value': self.value}


class OrdinalEncoder(TransformerMixin, BaseEstimator):
    """
    Encode specified ordinal variables with respect to target variable based on given statistic.
    Codes starts from 1 and unseen values or nan's are set to 0
    """

    def __init__(self, ordinal_feats=None, target_to_order=None, order_statistic=None):
        self.ordinal_feats = ordinal_feats
        self.target_to_order = target_to_order
        self.order_statistic = order_statistic

    def fit(self, X, y=None):
        """
        :param X: pandas DataFrame to find categorical values codes
        :return: self reference
        """
        self.codes = self.get_orders(X)
        return self

    def transform(self, y):
        """
        :param y: pandas DataFrame to replace with fitted code (apply encoding)
        :return: DataFrame with encoded ordinal features with respect to fitted order
        """
        if bool(self.codes):
            y = y.replace(to_replace=self.codes)
            y[self.ordinal_feats] = y[self.ordinal_feats].fillna(0)
        return y

    def get_orders(self, X):
        # todo: optimize get_orders
        codes = {}
        if self.ordinal_feats is not None:
            for feature in self.ordinal_feats:
                stat_values = self.target_to_order.groupby(X[feature]).describe(percentiles=[.5])
                stat_values = stat_values[self.order_statistic]
                sorted_values = stat_values.sort_values().axes[0].tolist()
                codes[feature] = {k: v for (k, v) in zip(sorted_values, range(1, len(sorted_values) + 1))}
        return codes

    def get_params(self, deep=True):
        return {
            'ordinal_feats': self.ordinal_feats,
            'target_to_order': self.target_to_order,
            'order_statistic': self.order_statistic
        }


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
    def __init__(self, feats_to_transform=None, func=lambda x: None):
        if feats_to_transform is None:
            feats_to_transform = []
        self.feats_to_transform = feats_to_transform
        self.func = func

    def fit(self, X, y=None):
        self.feats_to_transform = [feat for feat in self.feats_to_transform if feat in X.columns.tolist()]
        return self

    def transform(self, y):
        y[self.feats_to_transform] = y[self.feats_to_transform].apply(self.func)
        return y

    def get_params(self, deep=True):
        return {'feats_to_transform': self.feats_to_transform, 'func': self.func}
