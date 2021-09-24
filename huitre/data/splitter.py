import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from huitre import HuitreError


def split_data(df, split_method='fo', test_size=.2, random_state=42):
    """
    Method of splitting data into training data and test data
    :param df:
    :param split_method:
    :param test_size:
    :param random_state:
    :return:
    """
    if split_method == 'fo':
        train_set, test_set = _split_fo(df,
                                        test_size=test_size,
                                        random_state=random_state)
    elif split_method == 'tfo':
        train_set, test_set = _split_tfo(df, test_size=test_size)
    elif split_method == 'ufo':
        train_set, test_set = _split_ufo(df,
                                         test_size=test_size,
                                         random_state=random_state)
    elif split_method == 'utfo':
        train_set, test_set = _split_utfo(df, test_size=test_size)
    else:
        raise HuitreError('Invalid data_split value, expect: ufo, utfo')
    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)
    return train_set, test_set


def _split_fo(df, test_size=.2, random_state=42):
    train_set, test_set = train_test_split(df,
                                           test_size=test_size,
                                           random_state=random_state)
    return train_set, test_set


def _split_tfo(df, test_size=.2):
    df = df.sort_values(['timestamp']).reset_index(drop=True)
    split_idx = int(np.ceil(len(df) * (1 - test_size)))
    train_set, test_set = df.iloc[:split_idx, :].copy(), \
                          df.iloc[split_idx:, :].copy()
    return train_set, test_set


def _split_ufo(df, test_size=.2, random_state=42):
    """
    Split by ratio in user level
    :param df:
    :param test_size:
    :param random_state:
    :return:
    """
    train_set, test_set = pd.DataFrame(), pd.DataFrame()
    driver_ids = df['user']
    _, driver_indices = np.unique(np.array(driver_ids),
                                  return_inverse=True)
    gss = GroupShuffleSplit(n_splits=1,
                            test_size=test_size,
                            random_state=random_state)
    for train_idx, test_idx in gss.split(df, groups=driver_indices):
        train_set, test_set = df.loc[train_idx, :].copy(), df.loc[test_idx, :].copy()
    return train_set, test_set


def _split_utfo(df, test_size=.2):
    df = df.sort_values(['user', 'timestamp']).reset_index(drop=True)

    def time_split(grp):
        start_idx = grp.index[0]
        split_len = int(np.ceil(len(grp) * (1 - test_size)))
        split_idx = start_idx + split_len
        end_idx = grp.index[-1]

        return list(range(split_idx, end_idx + 1))

    test_index = df.groupby('user').apply(time_split).explode().values
    test_index = test_index[~pd.isnull(test_index)]
    test_set = df.loc[test_index, :]
    train_set = df[~df.index.isin(test_index)]
    return train_set, test_set
