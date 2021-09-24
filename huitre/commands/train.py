import os
import tensorflow as tf

from huitre import HuitreError, RANDOM_PREF_SAMPLING, \
    UNPOP_PREF_SAMPLING
from huitre.models import ModelFactory
from huitre.models.trainer import Trainer
from huitre.data import fetch_data, generator_factory
from huitre.logging import get_logger
from huitre.utils import mat_to_dict


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

    if not os.path.exists(training_params['model_dir']):
        os.makedirs(training_params['model_dir'], exist_ok=True)
    print(training_params['model_dir'])

    # fetch interaction matrices
    data = fetch_data(params)
    train_user_items = mat_to_dict(data['train_interactions'])
    valid_user_items = mat_to_dict(data['valid_interactions'])
    training_params['ref_user_items'] = {
        'train': train_user_items,
        'valid': valid_user_items
    }
    total_user_items = {k: v.union(valid_user_items[k])
                        for k, v in train_user_items.items()}
    # data generators
    generator_type = dataset_params.get('generator_type',
                                        'pairwise')
    max_num_prefs = training_params['model']['params'].get(
        'max_num_prefs', -1)
    train_generator = generator_factory(
        interactions=data['train_interactions'],
        batch_size=params['training']['batch_size'],
        num_negatives=params['training']['model']['params']['n_negatives'],
        random_state=dataset_params['random_state'],
        gen_type=generator_type,
        user_items=None,
        model_type=model_type,
        max_num_prefs=max_num_prefs,
        pref_sampling=pref_sampling)
    valid_generator = generator_factory(
        interactions=data['valid_interactions'],
        batch_size=params['training']['batch_size'],
        num_negatives=1,
        random_state=dataset_params['random_state'],
        gen_type=generator_type,
        user_items=total_user_items,
        model_type=model_type,
        max_num_prefs=max_num_prefs,
        pref_sampling=pref_sampling)
    # start model training
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    with tf.Session(config=sess_config) as sess:
        # create model
        model = ModelFactory.generate_model(sess=sess,
                                            params=training_params,
                                            n_users=data['n_users'],
                                            n_items=data['n_items'])
        sess.run(tf.global_variables_initializer())
        if model_type == 'transcf':
            train_generator.model = model
            valid_generator.model = model

        # create a trainer to train model
        trainer = Trainer(sess, model, params)

        logger.info('Start model training')
        trainer.fit(train_generator=train_generator,
                    valid_generator=valid_generator,
                    train_n_batches_per_epoch=train_generator.n_batches,
                    valid_n_batches_per_epoch=valid_generator.n_batches)
        logger.info('Model training done')
