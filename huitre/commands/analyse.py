import os
import pickle
import numpy as np
import tensorflow as tf

from huitre import HuitreError, RANDOM_PREF_SAMPLING, \
    UNPOP_PREF_SAMPLING
from huitre.utils import mat_to_dict, item_users_dict
from huitre.models import ModelFactory, Analyzer
from huitre.data import fetch_data
from huitre.logging import get_logger


def entrypoint(params):
    """ Command entrypoint.
    :param params: Deserialized JSON configuration file provided in CLI args.
    """
    logger = get_logger()
    tf.disable_eager_execution()

    dataset_params = params['dataset']
    training_params = params['training']
    model_params = training_params['model']

    dataset_spec = f'{dataset_params["name"]}_' \
                   f'{dataset_params["split_method"]}_' \
                   f'{dataset_params["n_core"]}-core'
    training_spec = f'lr{training_params["learning_rate"]}_' \
                    f'batch{training_params["batch_size"]}_' \
                    f'epoch{training_params["num_epochs"]}'
    model_type = model_params['type']
    pref_sampling = training_params.get('pref_sampling', 'random')
    pref_sampling = RANDOM_PREF_SAMPLING if pref_sampling == 'random' \
        else UNPOP_PREF_SAMPLING

    if model_type == 'cml':
        model_spec = f'{model_type}_' \
                     f'{training_spec}_' \
                     f'{model_params["params"]["n_negatives"]}negs_' \
                     f'margin{model_params["params"]["margin"]}'
    elif model_type == 'attcml':
        model_spec = f'{model_type}_' \
                     f'{training_spec}_' \
                     f'{model_params["params"]["n_negatives"]}negs_' \
                     f'margin{model_params["params"]["margin"]}_' \
                     f'beta{model_params["params"]["beta"]}_' \
                     f'nprefs{model_params["params"]["max_num_prefs"]}'
        if 'copy' in model_params['params'] and model_params['params']['copy'] is True:
            model_spec = f'{model_spec}_copyatt'
    elif model_type == 'lrml':
        model_spec = f'{model_type}_' \
                     f'{training_spec}_' \
                     f'{model_params["params"]["n_negatives"]}negs_' \
                     f'nram{model_params["params"]["num_memories"]}_' \
                     f'mode{model_params["params"]["memory_mode"]}_' \
                     f'margin{model_params["params"]["margin"]}'
        if 'copy' in model_params['params'] and model_params['params']['copy'] is True:
            model_spec = f'{model_spec}_copyram'
        if 'clip_ram' in model_params['params'] and model_params['params']['clip_ram'] is True:
            model_spec = f'{model_spec}_clipram'
    elif model_type == 'transcf':
        model_spec = f'{model_type}_' \
                     f'{training_spec}_' \
                     f'{model_params["params"]["n_negatives"]}negs_' \
                     f'margin{model_params["params"]["margin"]}_' \
                     f'regnbr{model_params["params"]["alpha_reg_nbr"]}_' \
                     f'regdist{model_params["params"]["alpha_reg_dist"]}'
    elif model_type == 'transh':
        model_spec = f'{model_type}_' \
                     f'{training_spec}_' \
                     f'{model_params["params"]["n_negatives"]}negs_' \
                     f'nrel{model_params["params"]["num_memories"]}_' \
                     f'mode{model_params["params"]["memory_mode"]}_' \
                     f'margin{model_params["params"]["margin"]}_' \
                     f'lambdareg{model_params["params"]["lambda_reg"]}'
        if 'clip_ram' in model_params['params'] and model_params['params']['clip_ram'] is True:
            model_spec = f'{model_spec}_clipram'
    elif model_type == 'hlre':
        model_spec = f'{model_type}_' \
                     f'{training_spec}_' \
                     f'{model_params["params"]["n_negatives"]}negs_' \
                     f'nrel{model_params["params"]["num_memories"]}_' \
                     f'mode{model_params["params"]["memory_mode"]}_' \
                     f'margin{model_params["params"]["margin"]}_' \
                     f'nprefs{model_params["params"]["max_num_prefs"]}_' \
                     f'beta{model_params["params"]["beta"]}'
        if 'copy' in model_params['params'] and model_params['params']['copy'] is True:
            model_spec = f'{model_spec}_copyram'
        if 'clip_ram' in model_params['params'] and model_params['params']['clip_ram'] is True:
            model_spec = f'{model_spec}_clipram'
        if 'catex' in model_params['params'] and model_params['params']['catex'] is not True:
            model_spec = f'{model_spec}_nocatex'
        if 'diff_ram' in model_params['params'] and model_params['params']['diff_ram'] is True:
            model_spec = f'{model_spec}_diffram'
        if pref_sampling != RANDOM_PREF_SAMPLING:
            model_spec = f'{model_spec}_unpop'
    elif model_type == 'hlrh':
        model_spec = f'{model_type}_' \
                     f'{training_spec}_' \
                     f'{model_params["params"]["n_negatives"]}negs_' \
                     f'nrel{model_params["params"]["num_memories"]}_' \
                     f'nnorm{model_params["params"]["num_norm_memories"]}_' \
                     f'mode{model_params["params"]["memory_mode"]}_' \
                     f'margin{model_params["params"]["margin"]}_' \
                     f'lambdareg{model_params["params"]["lambda_reg"]}_' \
                     f'nprefs{model_params["params"]["max_num_prefs"]}'
        if 'copy' in model_params['params'] and model_params['params']['copy'] is True:
            model_spec = f'{model_spec}_copyram'
        if 'clip_ram' in model_params['params'] and model_params['params']['clip_ram'] is True:
            model_spec = f'{model_spec}_clipram'
    elif model_type == 'jpte':
        model_spec = f'{model_type}_' \
                     f'{training_spec}_' \
                     f'{model_params["params"]["n_negatives"]}negs_' \
                     f'nrel{model_params["params"]["num_memories"]}_' \
                     f'mode{model_params["params"]["memory_mode"]}_' \
                     f'margin{model_params["params"]["margin"]}_' \
                     f'nprefs{model_params["params"]["max_num_prefs"]}'
        if 'copy' in model_params['params'] and model_params['params']['copy'] is True:
            model_spec = f'{model_spec}_copyram'
        if 'clip_ram' in model_params['params'] and model_params['params']['clip_ram'] is True:
            model_spec = f'{model_spec}_clipram'
        if pref_sampling != RANDOM_PREF_SAMPLING:
            model_spec = f'{model_spec}_unpop'
    else:
        raise HuitreError(f'Unknown model type {model_type}')

    training_params['model_dir'] = os.path.join(
        training_params['model_dir'],
        dataset_spec,
        model_spec)
    print(training_params['model_dir'])

    # fetch interaction matrices
    data = fetch_data(params)

    # get median number of interactions per user
    # logger.info(median_num_interactions(data))
    train_user_items = mat_to_dict(data['train_interactions'])
    valid_user_items = mat_to_dict(data['valid_interactions'])
    total_train_user_items = {k: v.union(valid_user_items[k])
                              for k, v in train_user_items.items()}
    k = params['analyse'].get('k', 3)
    user_metadata_path = os.path.join(params['cache']['path'], 'fo',
                                      f'users_meta_top{k}.pkl')
    item_metadata_path = os.path.join(params['cache']['path'], 'fo',
                                      f'items_meta_top{k}.pkl')
    test_interactions_path = os.path.join(
        params['cache']['path'], 'fo',
        f'test_interactions_has_meta_top{k}.pkl')
    categories_path = os.path.join(params['cache']['path'], 'fo',
                                   f'categories_top{k}.pkl')
    user_meta = pickle.load(open(user_metadata_path, 'rb'))
    item_meta = pickle.load(open(item_metadata_path, 'rb'))
    categories = pickle.load(open(categories_path, 'rb'))
    test_interactions = pickle.load(
        open(test_interactions_path, 'rb'))
    training_params['ref_user_items'] = {
        'train+valid': total_train_user_items,
        'user_meta': user_meta,
        'item_meta': item_meta,
        'test_interactions': test_interactions
    }
    logger.info(f'Number of users has metadata: {len(user_meta)}')
    logger.info(f'Number of items has metadata: {len(item_meta)}')
    logger.info(f'Number of categories: {len(categories)}')
    logger.info(list(categories)[:10])
    logger.info(f'Number of interactions has metadata: {len(test_interactions)}')

    params['extracted_relations_path'] = os.path.join('exp/extracted_relations',
                                                      dataset_params['name'],
                                                      model_spec)
    if not os.path.exists(params['extracted_relations_path']):
        os.makedirs(params['extracted_relations_path'], exist_ok=True)

    # session (TF1) - need to remove it for TF2 code later
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    with tf.Session(config=sess_config) as sess:
        # create model
        model = ModelFactory.generate_model(sess=sess,
                                            params=training_params,
                                            n_users=data['n_users'],
                                            n_items=data['n_items'],
                                            command='eval')
        analyzer = Analyzer(sess, model, params)
        analyzer.do_analyse()


def median_num_interactions(data):
    train_user_items = item_users_dict(data['train_interactions'])
    valid_user_items = item_users_dict(data['valid_interactions'])
    test_user_items = item_users_dict(data['test_interactions'])
    total_train_user_items = {k: v.union(valid_user_items[k])
                              for k, v in train_user_items.items()}
    total_user_items = {k: v.union(test_user_items[k])
                        for k, v in total_train_user_items.items()}
    num_interactions = [len(v) for _, v in total_user_items.items()]
    med_num_interactions = np.median(num_interactions)
    return med_num_interactions


def user_categories(data):
    train_user_items = mat_to_dict(data['train_interactions'])
    valid_user_items = mat_to_dict(data['valid_interactions'])
    test_user_items = mat_to_dict(data['test_interactions'])
    total_train_user_items = {k: v.union(valid_user_items[k])
                              for k, v in train_user_items.items()}
    total_user_items = {k: v.union(test_user_items[k])
                        for k, v in total_train_user_items.items()}
    item_ids = data['item_ids']
