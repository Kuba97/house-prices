import pandas as pd

from utils.make_dataset import load_split_dataset
from features.features_info import ID_COL, TARGET_COL
from .model_selection.select import grid_search_cv, cv_score
from utils.io_utils import pickle_file, MODEL_FOLDER


class Regressor:
    def __init__(self, model, target_transformer, cv_score_):
        self.model = model
        self.target_transformer = target_transformer
        self.cv_score = cv_score_

    @classmethod
    def as_validator(cls, target_transformer):
        return cls(None, target_transformer, None)

    def fit(self, x_train, y_train):
        y_train = self.target_transformer.transform(y_train)
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test, id_):
        predictions = pd.DataFrame(index=id_)
        y_pred = self.target_transformer.inverse_transform(self.model.predict(x_test))
        predictions[TARGET_COL] = y_pred
        return predictions

    def select_model(self, x_train, y_train, model, param_grid, set_best_estimator=False):
        y_train = self.target_transformer.transform(y_train)
        best_model, best_score = grid_search_cv(model, x_train, y_train, param_grid)
        if set_best_estimator:
            self.model = best_model
            self.cv_score = best_score

    def score_model(self, x_train, y_train):
        y_train = self.target_transformer(y_train)
        cv_score(self.model, x_train, y_train)

    def save(self, name=None):
        if name is None:
            name = type(self.model._final_estimator).__name__
        pickle_file(self, name, MODEL_FOLDER)
