from utils.make_dataset import load_split_dataset
from features.features_info import ID_COL, TARGET_COL
import models.estimators as pool
from models.regressor import Regressor
from features.transformers import LogTransform
from models.preprocessor import make_pipeline_with_estimator
from utils.submission import make_submission


def validate(regressor, **selection_kwargs):
    estimator = pool.xgboost
    model = make_pipeline_with_estimator(estimator)
    param_grid = {}  # fill with parameters to search within given values
    regressor.select_model(model=model, param_grid=param_grid, **selection_kwargs)


if __name__ == '__main__':
    x, y, _ = load_split_dataset('raw', 'train.csv', TARGET_COL, ID_COL)
    regressor = Regressor.as_validator(LogTransform())
    validate(regressor, x_train=x, y_train=y, set_best_estimator=False)
    # x, _, id_ = load_split_dataset('raw', 'test.csv', id_col=ID_COL)
    # make_submission(regressor, regressor.predict(x, id_))
