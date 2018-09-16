import os
import datetime
from data_io.io_utils import pickle_file, PATH_UP

REPORTS_FOLDER_NAME = 'reports'
FILENAME_DATE = "%Y-%m-%d"
COEFS_BASENAME = 'coefs'
PREDICTIONS_BASENAME = 'predicted'
FILENAME_SEPARATOR = '_'


def train_store_predictions(validator, x_train, y_train):
    validator.train(x_train, y_train)
    predicted = validator.predict(x_train)
    filename = PREDICTIONS_BASENAME + FILENAME_SEPARATOR + validator.get_model_name()
    save_report_file(predicted, filename)


def train_store_coefs(validator, x_train, y_train):
    coefs_df = validator.get_estimator_coefs(x_train, y_train)
    filename = COEFS_BASENAME + FILENAME_SEPARATOR + validator.get_model_name()
    save_report_file(coefs_df, filename)


def save_report_file(coefs_df, basic_name):
    filename = add_date_str(basic_name)
    report_path = os.path.join(os.getcwd(), PATH_UP, REPORTS_FOLDER_NAME)
    pickle_file(coefs_df, filename, report_path)


def add_date_str(basic_name):
    return basic_name + FILENAME_SEPARATOR + datetime.datetime.now().date().strftime(FILENAME_DATE)
