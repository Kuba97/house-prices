import numpy as np
import pandas as pd


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


def to_matrix(df):
    return df.values


def collapse_groups(df):
    group_dict = {'Blmngtn': 'f', 'Blueste': 'c', 'BrDale': 'a', 'BrkSide': 'a', 'ClearCr': 'f', 'CollgCr': 'f',
                  'Crawfor': 'f', 'Edwards': 'b', 'Gilbert': 'e', 'IDOTRR': 'a', 'MeadowV': 'a', 'Mitchel': 'd',
                  'NAmes': 'd', 'NPkVill': 'd', 'NWAmes': 'e', 'NridgHt': 'h', 'OldTown': 'b', 'SWISU': 'c',
                  'Sawyer': 'c', 'SawyerW': 'd', 'Somerst': 'g', 'StoneBr': 'h', 'Timber': 'h', 'Veenker': 'g'}
    df['Neighborhood'] = df['Neighborhood'].map(group_dict)
    return df
