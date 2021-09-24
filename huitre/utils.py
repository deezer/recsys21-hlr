from collections import defaultdict
import numpy as np
import scipy.sparse as sp

from huitre.logging import get_logger


def df_to_mat(df, n_rows, n_cols, binary=True):
    """
    Convert dataframe to matrix
    :param df:
    :param n_rows:
    :param n_cols:
    :param binary:
    :return:
    """
    dtype = np.int32 if binary is True else np.float32
    interactions_mat = sp.dok_matrix((n_rows, n_cols),
                                     dtype=dtype)
    interactions_mat[
        df.user.tolist(), df.item.tolist()] = 1
    interactions_mat = interactions_mat.tocsr()
    return interactions_mat


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


def item_users_dict(interactions):
    rows, cols = interactions.nonzero()
    res = defaultdict(set)
    for uid, iid in zip(rows, cols):
        res[iid].add(uid)
    return res


def add_mask(feature_mask, features, max_len):
    """
    Pendding mask at the end of features to have max_len
    :param feature_mask:
    :param features:
    :param max_len:
    :return:
    """
    for i in range(len(features)):
        if len(features[i]) < max_len:
            features[i] = features[i] + \
                          [feature_mask] * (max_len - len(features[i]))
    return np.array(features)


def get_item_popularity(interactions, max_count=None):
    interactions = sp.lil_matrix(interactions)
    popularity_dict = defaultdict(int)
    for uid, iids in enumerate(interactions.rows):
        for iid in iids:
            popularity_dict[iid] += 1
            if max_count is not None and popularity_dict[iid] > max_count:
                popularity_dict[iid] = max_count
    return popularity_dict


def best_cosine_similarities(relations):
    logger = get_logger()
    relations = np.array(relations)
    rel_norms = np.linalg.norm(relations, axis=1)
    n_relations = len(relations)
    bests = []
    for i in range(n_relations):
        similarities = np.ones(shape=relations.shape[0], dtype=np.float32)
        similarities = -1.0 * similarities
        for j in range(n_relations):
            if j != i:
                norm_i = rel_norms[i]
                norm_j = rel_norms[j]
                similarities[j] = (np.dot(relations[i], relations[j]) / (norm_i * norm_j))
        best_i = np.argmax(similarities)
        bests.append((i, best_i, similarities[best_i]))
        if i > 0 and i % 1000 == 0:
            logger.info(f'Finish processing for {i} relations')
    logger.info(f'Finish processing for {len(bests)} relations')
    return bests
