import os
import pandas as pd

PATH_UP = '..'
NAME_DATA_FOLDER = 'data'
PATH_DATA_FOLDER = os.path.join(os.getcwd(), PATH_UP, NAME_DATA_FOLDER)


def load_dataset(dataset_type, filename):
    dir_dataset = os.path.join(PATH_DATA_FOLDER, dataset_type, filename)
    return pd.read_csv(dir_dataset)


def load_split_dataset(dataset_type, filename, target_col=None, id_col=None):
    dataset = load_dataset(dataset_type, filename)
    x = dataset[dataset.columns.difference([id_col, target_col])]
    y = dataset[target_col] if target_col is not None else None
    id_ = dataset[id_col]
    return x, y, id_

