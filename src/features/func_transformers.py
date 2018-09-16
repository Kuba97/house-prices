import numpy as np


def log_transform(df, feats_to_log):
    for column in feats_to_log:
        df[column] = df[column].apply(lambda x: np.log(x) if x > 0 else 0)
    return df


def indicator(df, columns):
    df[columns] = (df[columns] > 0).astype(int)
    return df


def is_antique(df, antique_bound):
    new_col = (df['YearBuilt'] <= antique_bound).astype(int)
    df = df.assign(retro=new_col)
    return df


def to_categorical(df, feats_to_cat):
    for feat in feats_to_cat:
        df[feat] = df[feat].astype('str')
    return df
