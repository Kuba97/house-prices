from data.make_dataset import load_dataset
import pandas as pd
import numpy as np
from scipy import stats

COL_LABEL = 'SalePrice'
COL_ID = 'Id'
DROP_VAL_ONLY = ['Electrical']
LOG_TRANSFORM_COLS = ['SalePrice', 'GrLivArea', '1stFlrSF']

KIND_RAW = 'raw'
KIND_INTERIM = 'intermediate'
KIND_PRE = 'preprocessed'
NAME_RAW_TRAIN_SET = 'train.csv'
NAME_RAW_TEST_SET = 'test.csv'


def split_cols(dataset, label_col_name=COL_LABEL, id_col_name=COL_ID):
    x = dataset[dataset.columns.difference([label_col_name, id_col_name])]
    y = dataset[label_col_name]
    identifier = None if id_col_name is None else dataset[id_col_name]
    return x, y, identifier


def join_dfs(*args):
    total = pd.DataFrame()
    for df in args:
        total.append(df)
    return total


def center_data(df, cols):
    df[cols] = stats.zscore(df[cols])


def get_nan_info(df):
    nans_mean = df.isnull().mean()
    return nans_mean[nans_mean > 0]


def deal_with_nan(dataset):
    nan_info = get_nan_info(dataset)
    nan_cols = nan_info.axes[0].tolist()
    cols_to_drop = [x for x in nan_cols if x not in DROP_VAL_ONLY]
    dataset = dataset[dataset.columns.difference(cols_to_drop)]
    dataset = dataset.dropna()
    if dataset.isnull().sum().max() > 0:
        raise Exception('NaN still in dataset')
    return dataset


def log_transform(dataset, cols):
    for col in cols:
        dataset[col] = np.log(dataset[col])


def check_for_nan(df):
    nan_info = df.isnull().sum()
    if nan_info.max() > 0:
        raise ValueError('Dataset contains nan values: ', nan_info[nan_info > 0])


def preprocess_procedure(is_test=False):
    dataset = load_dataset(KIND_RAW, NAME_RAW_TRAIN_SET)
    # dataset = deal_with_nan(dataset)
    log_transform(dataset, LOG_TRANSFORM_COLS)
    x, y, ids = split_cols(dataset)
    x = pd.get_dummies(x, dummy_na=True)
    check_for_nan(x)
    return x.values, y.values, ids.values
