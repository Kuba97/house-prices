import os
import datetime
import pickle as pkl
import pandas as pd

PROJECT_PATH = os.getcwd()
PATH_UP = '..'
DATA_FOLDER = 'data'
SUBMISSION_FOLDER = 'submissions'
FITTED_MODEL_FOLDER = 'fitted_models'
REPORT_FOLDER = 'reports'
SUBMISSION_PATH = os.path.join(PROJECT_PATH, PATH_UP, SUBMISSION_FOLDER)
FITTED_MODEL_PATH = os.path.join(PROJECT_PATH, PATH_UP, FITTED_MODEL_FOLDER)
REPORT_PATH = os.path.join(PROJECT_PATH, PATH_UP, REPORT_FOLDER)
FILENAME_SEPARATOR = '_'

SUBMISSION_FORMAT = REPORT_FORMAT = '.csv'
DUMP_FORMAT = '.pkl'

def pickle_file(obj, filename, path):
    path_with_name = os.path.join(path, filename)
    with open(path_with_name, 'wb') as file:
        pkl.dump(obj, file, protocol=pkl.HIGHEST_PROTOCOL)


def unpickle_file(path):
    with open(path, 'rb') as file:
        loaded = pkl.load(file)
    return loaded


def save_csv(df, filename, path):
    path_with_name = os.path.join(path, filename)
    df.to_csv(path_with_name, header=True)


def load_csv(path):
    return pd.read_csv(path)


def tag_filename_with_datetime(raw_filename, file_format, additional_info=''):
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    tagged_filename = date + FILENAME_SEPARATOR + raw_filename + FILENAME_SEPARATOR + additional_info + file_format
    return tagged_filename
