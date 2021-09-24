import os
from collections import defaultdict
import numpy as np
import tensorflow as tf

from huitre import HuitreError, RANDOM_PREF_SAMPLING, \
    UNPOP_PREF_SAMPLING
from huitre.utils import mat_to_dict, get_item_popularity
from huitre.models import ModelFactory, Recommender
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
    elif model_type == 'bpr':
        model_spec = f'{model_type}_' \
                     f'{training_spec}_' \
                     f'ureg{model_params["params"]["user_regularization"]}_' \
                     f'ubias{model_params["params"]["user_bias_regularization"]}_' \
                     f'preg{model_params["params"]["pos_item_regularization"]}_' \
                     f'nreg{model_params["params"]["neg_item_regularization"]}_' \
                     f'ibias{model_params["params"]["item_bias_regularization"]}_' \
                     f'{model_params["params"]["n_negatives"]}negs'
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
    params['eval']['reco_dir'] = os.path.join(
        params['eval']['reco_dir'],
        dataset_params["name"],
        model_spec)
    if not os.path.exists(params['eval']['reco_dir']):
        os.makedirs(params['eval']['reco_dir'], exist_ok=True)
    # fetch interaction matrices
    data = fetch_data(params)
    train_user_items = mat_to_dict(data['train_interactions'])
    valid_user_items = mat_to_dict(data['valid_interactions'])
    test_user_items = mat_to_dict(data['test_interactions'])
    total_train_user_items = {k: v.union(valid_user_items[k])
                              for k, v in train_user_items.items()}
    training_params['ref_user_items'] = {
        'train': train_user_items,
        'valid': valid_user_items,
        'train+valid': total_train_user_items,
        'test': test_user_items,
    }
    params['eval']['items_metadata'] = {
        'popularity': get_item_popularity(data['train_interactions'])
    }

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
        # create a recommender to evaluate model or
        # to make recommendations
        recommender = Recommender(sess, model, params)
        all_scores = recommender.eval(corpus='test')
        if 'acc' not in params['eval'] or \
                params['eval']['acc'] is True:
            final_scores = defaultdict(list)
            for scores in all_scores:
                for k, v in scores.items():
                    if type(v) is tuple:
                        final_scores[f'local-{k}'].append(v[0])
                        final_scores[f'global-{k}'].append(v[1])
                    else:
                        final_scores[k].append(v)
            mean_scores = {k: np.mean(v) for k, v in final_scores.items()}
            std_scores = {k: np.std(v) for k, v in final_scores.items()}
            scores_message = [f'{k} {mean_scores[k]:6.4f} +/- {std_scores[k]:6.4f}'
                              for k, _ in final_scores.items()]
            scores_message = ', '.join(scores_message)
        else:
            print(all_scores)
            scores_message = f'MMR: {np.mean(all_scores)} +/- {np.std(all_scores)}'
        logger.info(f'Finish eval: {scores_message}')
