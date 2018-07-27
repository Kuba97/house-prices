import os
import pandas as pd
import numpy as np

DATA_FOLDER = 'data'
PROJECT_FOLDER = os.path.join(os.getcwd(), '..')
DATA_FOLDER_DIR = os.path.join(PROJECT_FOLDER, DATA_FOLDER)

TRAIN_FILE = 'train_prepared.csv'
TRAIN_FILE_DIR = os.path.join(DATA_FOLDER_DIR, TRAIN_FILE)


def load(train_file_dir, label_col_name, id_col_name):
    raw_data = pd.read_csv(train_file_dir)
    x_train = raw_data[raw_data.columns.difference([label_col_name, id_col_name])]
    y_train = raw_data[label_col_name]
    ids_train = raw_data[id_col_name]
    return x_train, y_train, ids_train


class Data:
    def __init__(self, train_file_dir=TRAIN_FILE_DIR, label_col_name=None, id_col_name=None):
        self.x_train, self.y_train, self.ids = load(train_file_dir, label_col_name, id_col_name)

    def add_intercept(self):
        if isinstance(self.x_train, pd.DataFrame):
            self.x_train.insert(loc=0, column='intercept', value=1)
        elif isinstance(self.x_train, np.ndarray):
            new_x_train = np.ones([self.x_train.shape[0], self.x_train.shape[1] + 1])
            new_x_train[:, 1:] = self.x_train
            self.x_train = new_x_train

    def to_matrix(self):
        self.x_train = self.x_train.values
        self.y_train = self.y_train.values
        self.ids = self.ids.values

    def shuffle(self):
        indices = np.arange(len(self.x_train))
        np.random.shuffle(indices)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]
        self.ids = self.ids[indices]
