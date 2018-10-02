import datetime
from utils.io_utils import FILENAME_SEPARATOR, SUBMISSION_PATH, save_csv
SUBMISSION_FORMAT = '.csv'


def make_submission(regressor, predictions):
    save_csv(predictions, make_submission_filename(regressor), SUBMISSION_PATH)


def make_submission_filename(regressor):
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    regressor_name = type(regressor.model._final_estimator).__name__
    cv_info = 'cv-{:.3f}'.format(regressor.cv_score)
    file_format = '.csv'
    filename = date + FILENAME_SEPARATOR + regressor_name + FILENAME_SEPARATOR + cv_info + file_format
    return filename
