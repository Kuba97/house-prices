from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV

DEF_SCORING = 'neg_mean_squared_error'
DEF_CV_FOLDS = 5
DEF_NJOBS = 1


class Validator:
    def __init__(self, preprocessor, estimator):
        self.preprocessor = preprocessor
        self.estimator = estimator

    def train(self, x_train, y_train):
        x_prep = self.preprocessor.fit_transform(x_train)
        self.estimator = self.estimator()
        self.estimator.fit(x_prep, y_train)

    def predict(self, x):
        return self.estimator.predict(self.preprocessor.transform(x))

    def get_cross_val_scores(self, x_train, y_train, scoring_name=DEF_SCORING, cv_folds=DEF_CV_FOLDS):
        joined_pipeline = self._join_pipelines()
        scores = cross_val_score(joined_pipeline, x_train, y_train, scoring=scoring_name, cv=cv_folds, n_jobs=DEF_NJOBS)
        return scores

    def grid_search(self, x_train, y_train, param_grid, scoring_name=DEF_SCORING, cv_folds=DEF_CV_FOLDS,
                    n_jobs=DEF_NJOBS):
        gs_model = GridSearchCV(
            estimator=self._join_pipelines(),
            param_grid=param_grid,
            scoring=scoring_name,
            cv=cv_folds,
            n_jobs=n_jobs
        )
        gs_model.fit(x_train, y_train)
        return gs_model

    def get_estimator_coefs(self, x_train, y_train):
        # todo: final estimator coefs retriever
        # todo: feature names from preprocessing pipeline retriever
        # todo: bind together into DataFrame names and coefs
        pass

    def get_model_name(self):
        return self.estimator.__name__

    def _join_pipelines(self):
        return make_pipeline(self.preprocessor, self.estimator)
