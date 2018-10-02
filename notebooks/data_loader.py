import os
import pandas as pd

DIR_UP = '..'
NAME_DATAFOLDER = 'data'

DIR_DATA_FOLDER = os.path.join(os.getcwd(), DIR_UP, NAME_DATAFOLDER)

NAME_RAW_DATAFOLDER = 'raw'
NAME_RAW_TRAIN_FILE = 'train.csv'
NAME_RAW_TEST_FILE = 'test.csv'

DIR_RAW_TRAIN = os.path.join(DIR_DATA_FOLDER, NAME_RAW_DATAFOLDER, NAME_RAW_TRAIN_FILE)
DIR_RAW_TEST = os.path.join(DIR_DATA_FOLDER, NAME_RAW_DATAFOLDER, NAME_RAW_TEST_FILE)

def load_data(filepath=DIR_RAW_TRAIN):
    return pd.read_csv(filepath)
