from .io_utils import FILENAME_SEPARATOR, SUBMISSION_PATH, SUBMISSION_FORMAT, save_csv, tag_filename_with_datetime


def make_submission(regressor, predictions):
    save_csv(predictions, make_submission_filename(regressor), SUBMISSION_PATH)


def make_submission_filename(regressor):
    regressor_name = type(regressor.model._final_estimator).__name__
    cv_info = 'cv-{:.3f}'.format(regressor.cv_score)
    return tag_filename_with_datetime(regressor_name, SUBMISSION_FORMAT, cv_info)
