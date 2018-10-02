import os
import pickle as pkl
import pandas as pd

PROJECT_PATH = os.getcwd()
PATH_UP = '..'
DATA_FOLDER = 'data'
SUBMISSION_FOLDER = 'submissions'
MODEL_FOLDER = 'fitted_models'
SUBMISSION_PATH = os.path.join(PROJECT_PATH, PATH_UP, SUBMISSION_FOLDER)
MODEL_PATH = os.path.join(PROJECT_PATH, PATH_UP, MODEL_FOLDER)
FILENAME_SEPARATOR = '_'


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
