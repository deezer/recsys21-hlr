import os
import numpy as np
import scipy.sparse as sp
from huitre import HuitreError
from huitre.data.splitter import split_data
from huitre.utils import df_to_mat
from huitre.logging import get_logger

from huitre.data.datasets import csv_reader, json_reader


def fetch_data(params):
    """
    Fetch movielens data
    :param params: dataset parameters
    :return:
    """
    cache_params = params['cache']
    dataset_params = params['dataset']

    cache_path = cache_params['path']
    train_interactions_name = cache_params['train_interactions']
    valid_interactions_name = cache_params['valid_interactions']
    test_interactions_name = cache_params['test_interactions']
    entities_name = cache_params['entities']
    split_method = dataset_params.get('split_method', 'fo')

    split_path = os.path.join(cache_path, split_method)
    train_interactions_path = os.path.join(split_path,
                                           f'{train_interactions_name}.npz')
    valid_interactions_path = os.path.join(split_path,
                                           f'{valid_interactions_name}.npz')
    test_interactions_path = os.path.join(split_path,
                                          f'{test_interactions_name}.npz')
    entities_path = os.path.join(split_path,
                                 f'{entities_name}.npz')
    logger = get_logger()
    if not os.path.exists(train_interactions_path) or not \
        os.path.exists(valid_interactions_path) or not \
        os.path.exists(test_interactions_path) or not \
        os.path.exists(entities_path):

        if not os.path.exists(split_path):
            os.mkdir(split_path)

        # fetch original interactions
        data = _fetch_interactions(dataset_params)

        n_core = dataset_params.get('n_core', 1)
        logger.debug(f'Preprocessing for {dataset_params["name"]} {n_core}-core')
        if n_core > 1:
            data, user_ids, item_ids = _preprocessing(data, n_core)
        else:
            data = data.rename(columns={'org_user': 'user',
                                        'org_item': 'item'})
            user_ids = data.user.unique()
            item_ids = data.item.unique()

        # split data
        logger.debug('Split data into train/val/test')
        train_set, test_set = split_data(data,
                                         split_method=split_method,
                                         test_size=dataset_params['test_size'],
                                         random_state=dataset_params['random_state'])
        train_set, val_set = split_data(train_set,
                                        split_method=split_method,
                                        test_size=dataset_params['val_size'],
                                        random_state=dataset_params['random_state'])
        # convert to sparse and save
        binary = dataset_params.get('binary', True)
        train_interactions = df_to_mat(train_set,
                                       n_rows=len(user_ids),
                                       n_cols=len(item_ids),
                                       binary=binary)
        valid_interactions = df_to_mat(val_set,
                                       n_rows=len(user_ids),
                                       n_cols=len(item_ids),
                                       binary=binary)
        test_interactions = df_to_mat(test_set,
                                      n_rows=len(user_ids),
                                      n_cols=len(item_ids),
                                      binary=binary)
        # save to file
        logger.debug('Save data to cache')
        sp.save_npz(train_interactions_path, train_interactions)
        sp.save_npz(valid_interactions_path, valid_interactions)
        sp.save_npz(test_interactions_path, test_interactions)
        np.savez(entities_path, user_ids=user_ids, item_ids=item_ids)
    else:
        logger.info('Load interaction matrices')
        train_interactions = sp.load_npz(train_interactions_path)
        valid_interactions = sp.load_npz(valid_interactions_path)
        test_interactions = sp.load_npz(test_interactions_path)
        entity_ids = np.load(entities_path, allow_pickle=True)
        user_ids = entity_ids['user_ids']
        item_ids = entity_ids['item_ids']
    logger.info(f'Num users: {len(user_ids)}')
    logger.info(f'Num items: {len(item_ids)}')
    logger.info(f'Num of train interactions: '
                f'{train_interactions.count_nonzero()}')
    logger.info(f'Num of valid interactions: '
                f'{valid_interactions.count_nonzero()}')
    logger.info(f'Num of test interactions: '
                f'{test_interactions.count_nonzero()}')

    return {
        'train_interactions': train_interactions,
        'valid_interactions': valid_interactions,
        'test_interactions': test_interactions,
        'user_ids': user_ids,
        'item_ids': item_ids,
        'n_users': train_interactions.shape[0],
        'n_items': train_interactions.shape[1]
    }


def _fetch_interactions(params):
    """
    Fetch interactions from file
    :param params: dataset parameters
    """
    if params['file_format'] == 'csv':
        data = csv_reader.read_data(params)
    elif params['file_format'] == 'json':
        data = json_reader.read_data(params)
    else:
        raise HuitreError(f'Do not support {params["file_format"]}')
    # set rating >= threshold as positive samples
    pos_threshold = params.get('pos_threshold', None)
    if pos_threshold is not None and pos_threshold > 0:
        data = data[data['rating'] >= pos_threshold].\
            reset_index(drop=True)
    if 'binary' in params and params['binary'] is True:
        data['rating'] = 1
    return data


def _preprocessing(data, n_core):
    """
    Preprocessing data to get n-core dataset.
    Each user has at least n items in his preference
    Each item is interacted by at least n users
    :param data:
    :param n_core:
    :return:
    """
    get_logger().debug(f'Get {n_core}-core data')

    def filter_user(df):
        tmp = df.groupby(['org_user'], as_index=False)['org_item'].count()
        tmp.rename(columns={'org_item': 'cnt_item'},
                   inplace=True)
        df = df.merge(tmp, on=['org_user'])
        df = df[df['cnt_item'] >= n_core].reset_index(drop=True).copy()
        df.drop(['cnt_item'], axis=1, inplace=True)
        return df

    def filter_item(df):
        tmp = df.groupby(['org_item'], as_index=False)['org_user'].count()
        tmp.rename(columns={'org_user': 'cnt_user'},
                   inplace=True)
        df = df.merge(tmp, on=['org_item'])
        df = df[df['cnt_user'] >= n_core].reset_index(drop=True).copy()
        df.drop(['cnt_user'], axis=1, inplace=True)
        return df

    while 1:
        data = filter_user(data)
        data = filter_item(data)
        chk_u = data.groupby('org_user')['org_item'].count()
        chk_i = data.groupby('org_item')['org_user'].count()
        if len(chk_i[chk_i < n_core]) <= 0 and len(chk_u[chk_u < n_core]) <= 0:
            break

    data = data.dropna()
    user_ids = data.org_user.unique()
    item_ids = data.org_item.unique()
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}
    data.loc[:, 'user'] = data.org_user.apply(lambda x: user_id_map[x])
    data.loc[:, 'item'] = data.org_item.apply(lambda x: item_id_map[x])

    return data, user_ids, item_ids
