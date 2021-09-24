from collections import defaultdict
import numpy as np

from huitre import RANDOM_PREF_SAMPLING
from huitre.logging import get_logger
from huitre.data.sampler import Sampler
from huitre.utils import mat_to_dict, item_users_dict, add_mask


class BatchGenerator:
    """
    Batch Generator is responsible for train/valid
    batch data generation
    """
    def __init__(self, interactions, batch_size,
                 num_negatives, random_state=42,
                 user_items=None,
                 **kwargs):
        """
        Initialize a data generator
        :param interactions:
        :param batch_size:
        :param num_negatives:
        :param random_state:
        :param user_items: default user items
        :param kwargs:
        """
        self.logger = get_logger()
        self.interactions = interactions
        if user_items is None:
            self.user_items = mat_to_dict(self.interactions,
                                          criteria=None)
        else:
            self.user_items = user_items
        self.batch_size = batch_size
        self.random_state = random_state
        n_interactions = self.interactions.count_nonzero()
        self.n_batches = int(n_interactions / self.batch_size)
        if self.n_batches * self.batch_size < n_interactions:
            self.n_batches += 1
        self.current_batch_idx = 0
        if kwargs is not None:
            self.__dict__.update(kwargs)
        # positive user item pairs
        self.user_pos_item_pairs = np.asarray(self.interactions.nonzero()).T
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(self.user_pos_item_pairs)
        self.sampler = Sampler(user_items=self.user_items,
                               n_items=self.interactions.shape[1],
                               n_negatives=num_negatives)
        self.ground_user_items = None
        self.ground_item_users = None
        if self.model_type == 'transcf':
            if user_items is None:
                self.ground_user_items = mat_to_dict(interactions)
                self.ground_item_users = item_users_dict(interactions)
            else:
                self.ground_user_items = user_items
                self.ground_item_users = defaultdict(set)
                for uid, iids in user_items.items():
                    for iid in iids:
                        self.ground_item_users[iid].add(uid)
        if self.max_num_prefs > 0:
            self.pref_items = self._get_pref_items(interactions,
                                                   user_items)
            if self.pref_sampling != RANDOM_PREF_SAMPLING:
                self.logger.info('SAMPLING PREFS BASED ON ITEM RANKS')
                self.item_ranks = defaultdict(int)
                for _, iids in self.pref_items.items():
                    for iid in iids:
                        self.item_ranks[iid] += 1

    def next_batch(self):
        """
        Batch generator.
        :return:
        """
        if self.current_batch_idx == self.n_batches:
            self.current_batch_idx = 0

        batch_samples = self._batch_sampling(self.current_batch_idx)
        self.current_batch_idx += 1
        return batch_samples

    def _batch_sampling(self, batch_index):
        """
        Batch generation based on a specific sampling strategy
        :param batch_index:
        :return:
        """
        """
        Batch generation based on a specific sampling strategy
        :param batch_index:
        :return:
        """
        batch_user_ids, batch_pos_ids, batch_neg_ids = \
            self._batch_triplets(batch_index)

        if self.model_type != 'transcf':
            if self.max_num_prefs <= 0:
                return batch_user_ids, batch_pos_ids, batch_neg_ids
            else:
                batch_pref_ids = []
                for uid, iid in zip(batch_user_ids, batch_pos_ids):
                    prefs = list(self.pref_items[uid].difference({iid}))
                    if len(prefs) > self.max_num_prefs:
                        if self.pref_sampling == RANDOM_PREF_SAMPLING:
                            prefs = np.random.choice(prefs,
                                                     size=self.max_num_prefs,
                                                     replace=False)
                        else:
                            item_pops = [1. / self.item_ranks[iid] for iid in prefs]
                            item_sum = np.sum(item_pops)
                            item_pops = [pop / item_sum for pop in item_pops]
                            prefs = np.random.choice(prefs,
                                                     size=self.max_num_prefs,
                                                     p=item_pops)
                    batch_pref_ids.append(prefs)
                batch_num_prefs = np.array([len(pref) for pref in batch_pref_ids])
                curr_max_num_prefs = min(self.max_num_prefs,
                                         np.max(batch_num_prefs))
                batch_pref_ids = add_mask(
                    self.interactions.shape[1],
                    batch_pref_ids,
                    curr_max_num_prefs)
                return batch_user_ids, batch_pos_ids, batch_neg_ids, \
                       batch_pref_ids, batch_num_prefs

        # if model is transcf, generate more data
        user_embeddings, item_embeddings = self.model.embeddings()
        user_neighbors = self._neighbor_embedding(batch_user_ids,
                                                  self.ground_user_items,
                                                  item_embeddings)
        pos_neighbors = self._neighbor_embedding(batch_pos_ids,
                                                 self.ground_item_users,
                                                 user_embeddings)
        neg_neighbors = self._neighbor_embedding(batch_neg_ids.flatten(),
                                                 self.ground_item_users,
                                                 user_embeddings)
        return batch_user_ids, batch_pos_ids, batch_neg_ids, \
               user_neighbors, pos_neighbors, neg_neighbors

    def _batch_triplets(self, batch_index):
        """
        Generate triplets (user, pos_id, neg_ids)
        :param batch_index:
        :return:
        """
        # user_ids, pos_ids
        batch_user_pos_items_pairs = self.user_pos_item_pairs[
                                     batch_index * self.batch_size:
                                     (batch_index + 1) * self.batch_size, :]
        # batch user_ids
        batch_user_ids = np.array(
            [uid for uid, _ in batch_user_pos_items_pairs])
        # batch positive item_ids
        batch_pos_ids = np.array([iid for _, iid in batch_user_pos_items_pairs])
        # batch negative item_ids
        batch_neg_ids = self.sampler.sampling(batch_user_ids)
        return batch_user_ids, batch_pos_ids, batch_neg_ids

    @classmethod
    def _neighbor_embedding(cls, ids, ground_dict, embedding):
        """
        Get neighbor embedding
        :param ids:
        :param ground_dict:
        :param embedding:
        :return:
        """
        neighbor_ids = [ground_dict[eid] for eid in ids]
        res_embeddings = []
        for nids in neighbor_ids:
            neighbor_embeddings = embedding[list(nids), :]
            res_embeddings.append(np.mean(neighbor_embeddings, axis=0))
        return np.array(res_embeddings)

    @classmethod
    def _get_pref_items(cls, interactions, user_items):
        """
        Get favorite items in user preferences
        :param interactions:
        :param user_items:
        :return:
        """
        # get ground truth user items
        ground_user_items = mat_to_dict(interactions)
        if user_items is None:
            # in the case training, pref items are
            # the same as user items
            pref_items = ground_user_items
        else:
            # in the case validation, pref items are the one
            # in training data, not in current interactions
            pref_items = {
                uid: items.difference(ground_user_items[uid])
                if uid in ground_user_items else items
                for uid, items in user_items.items()}
        return pref_items
