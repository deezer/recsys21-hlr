import os
import json
import numpy as np
import scipy.sparse as sp
import pickle
from collections import defaultdict
import implicit

from huitre.evaluators import Evaluator
from huitre.utils import get_item_popularity


def load_configuration(descriptor):
    """
    Load configuration from the given descriptor.
    Args:
        descriptor:
    Returns:

    """
    if not os.path.exists(descriptor):
        raise IOError(f'Configuration file {descriptor} not found')
    with open(descriptor, 'r') as stream:
        return json.load(stream)


def mat_to_dict(interactions, criteria=None):
    """
    Convert sparse matrix to dictionary of set
    :param interactions: scipy sparse matrix
    :param criteria:
    :return:
    """
    if not sp.isspmatrix_lil(interactions):
        interactions = sp.lil_matrix(interactions)
    n_rows = interactions.shape[0]
    res = {
        u: set(interactions.rows[u]) for u in range(n_rows)
        if criteria is None or
           (criteria is not None and criteria(interactions, u) is True)
    }
    return res


def load_data(params):
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
    train_interactions = sp.load_npz(train_interactions_path)
    valid_interactions = sp.load_npz(valid_interactions_path)
    test_interactions = sp.load_npz(test_interactions_path)
    entity_ids = np.load(entities_path, allow_pickle=True)
    user_ids = entity_ids['user_ids']
    item_ids = entity_ids['item_ids']

    print(f'Num users: {len(user_ids)}')
    print(f'Num items: {len(item_ids)}')
    print(f'Num of train interactions: '
                f'{train_interactions.count_nonzero()}')
    print(f'Num of valid interactions: '
                f'{valid_interactions.count_nonzero()}')
    print(f'Num of test interactions: '
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


def recommend(mdl, train_interactions, test_user_items, random_state,
              n_cohort, reco_dir, n_eval_users, n_rec_items):
    results = []
    user_ids = test_user_items.keys()
    # remove users that do not have any interactions in test
    user_ids = [uid for uid in user_ids if len(test_user_items[uid]) > 0]
    rng = np.random.RandomState(random_state)
    rng.shuffle(user_ids)
    for eval_idx in range(n_cohort):
        cohort_cache = os.path.join(reco_dir,
                                    f'cohort_{eval_idx + 1}.pkl')
        print(f'*********** Eval num: {eval_idx + 1} for '
              f'{n_eval_users} users ***********')
        if not os.path.exists(cohort_cache):
            reco_items = dict()
            eval_user_ids = user_ids[eval_idx * n_eval_users:
                                     (eval_idx + 1) * n_eval_users]
            for idx, uid in enumerate(eval_user_ids):
                reco_items[uid] = mdl.recommend(uid, train_interactions,
                                                N=n_rec_items)
                if (idx + 1) % 500 == 0:
                    print(f'Finish recommendation for {idx + 1} users')
            pickle.dump(reco_items, open(cohort_cache, 'wb'))
        else:
            reco_items = pickle.load(open(cohort_cache, 'rb'))
        results.append(reco_items)
    return results


def reco_items(recommendation):
    res = dict()
    for uid, items in recommendation.items():
        res[uid] = [iid for iid, _ in items]
    return res


# load data
config_file = 'configs/echonest/10-core/als/fo.json'
# config_file = 'configs/mvlens/10-core/als/fo.json'
# config_file = 'configs/yelp/5-core/als/fo.json'
# config_file = 'configs/amzb/10-core/als/fo.json'

params = load_configuration(config_file)
data = load_data(params)

train_item_users = data['train_interactions'].T

regularization = params['training']['model']['params']['regularization']
num_epochs = params['training']['num_epochs']
dataname = params['dataset']['name']
n_core = params['dataset']['n_core']
model_path = f'exp/model/als/{dataname}_fo_{n_core}-core/' \
             f'als_reg{regularization}_nepoch{num_epochs}'
reco_path = f'exp/reco/als/{dataname}_fo_{n_core}-core/als_reg{regularization}_' \
            f'nepoch{num_epochs}'
if not os.path.exists(reco_path):
    os.makedirs(reco_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)
model_path = f'{model_path}/model.pkl'

# initialize a model
if not os.path.exists(model_path):
    model = implicit.als.AlternatingLeastSquares(factors=params['training']['embedding_dim'],
                                                 iterations=params['training']['num_epochs'],
                                                 regularization=regularization,
                                                 calculate_training_loss=True,
                                                 use_gpu=False)
    model.fit(train_item_users)
    pickle.dump(model, open(model_path, "wb"))
else:
    model = pickle.load(open(model_path, "rb"))

test_user_items = mat_to_dict(data['test_interactions'])
recommendations = recommend(model,
                            train_interactions=data['train_interactions'],
                            test_user_items=test_user_items,
                            random_state=params['dataset']['random_state'],
                            n_cohort=params['eval']['n_cohort'],
                            reco_dir=reco_path,
                            n_eval_users=params['eval']['n_users'],
                            n_rec_items=10)

items_metadata = {
    'popularity': get_item_popularity(data['train_interactions'])
}

evaluator = Evaluator(params['eval'],
                      ref_user_items=test_user_items)

all_scores = []
for i in range(params['eval']['n_cohort']):
    if 'acc' not in params['eval'] or params['eval']['acc'] is True:
        scores = evaluator.eval(reco_items(recommendations[i]))
    else:
        scores = evaluator.eval_non_acc(reco_items(recommendations[i]),
                                        items_metadata)
    all_scores.append(scores)

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

print(f'Finish eval: {scores_message}')
