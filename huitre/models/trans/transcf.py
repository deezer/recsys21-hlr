from collections import defaultdict
import numpy as np
import tensorflow as tf
from huitre.models.model import Model


class TransCF(Model):
    """
    Collaborative Translational Metric Learning
    :reference
    """
    def __init__(self, sess, params, n_users, n_items):
        super(TransCF, self).__init__(sess, params, n_users, n_items)
        self.alpha_reg_nbr = params['model']['params'].get(
            'alpha_reg_nbr', 0.1)
        self.alpha_reg_dist = params['model']['params'].get(
            'alpha_reg_dist', 0.1)
        self.ref_user_items = params.get('ref_user_items', None)
        if self.ref_user_items is not None:
            self.ref_user_items = self.ref_user_items['train']
            self.ref_item_users = defaultdict(set)
            for uid, iids in self.ref_user_items.items():
                for iid in iids:
                    self.ref_item_users[iid].add(uid)
        self.dense_user_neighbors = None
        self.dense_item_neighbors = None
        self.score_user_ids_dict =  None

    def build_ui_relations(self, user_ids,
                           user_embeddings,
                           item_embeddings):
        self.dense_user_neighbors = self._neighbor_embedding(
            user_ids,
            self.ref_user_items,
            item_embeddings)
        self.dense_item_neighbors = self._neighbor_embedding(
            range(self.n_items),
            self.ref_item_users,
            user_embeddings)
        self.score_user_ids_dict = {uid: idx for idx, uid in enumerate(user_ids)}

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

    def _create_placeholders(self):
        super(TransCF, self)._create_placeholders()
        self.logger.debug('--> Create neighbor placeholders')
        self.user_neighbors = tf.compat.v1.placeholder(
            name='user_neighbors',
            dtype=tf.float32,
            shape=[None, self.embedding_dim])
        self.pos_neighbors = tf.compat.v1.placeholder(
            name='pos_neighbors',
            dtype=tf.float32,
            shape=[None, self.embedding_dim])
        self.neg_neighbors = tf.compat.v1.placeholder(
            name='neg_neighbors',
            dtype=tf.float32,
            shape=[None, self.embedding_dim])
        self.scored_relations = tf.compat.v1.placeholder(
            name='neg_neighbors',
            dtype=tf.float32,
            shape=[None, self.n_items, self.embedding_dim])

    def build_feedict(self, batch):
        feed_dict = super(TransCF, self).build_feedict(batch)
        feed_dict[self.user_neighbors] = batch[3]
        feed_dict[self.pos_neighbors] = batch[4]
        feed_dict[self.neg_neighbors] = batch[5]
        return feed_dict

    def _create_loss(self):
        super(TransCF, self)._create_loss()
        self.reg_nbr = self._neighbor_regularization()
        self.reg_dist = self._dist_regularization()
        self.loss = self.loss + self.reg_nbr + self.reg_dist

    def _neighbor_regularization(self):
        user_nbr_reg = tf.reduce_sum(
            tf.squared_difference(self.u_vectors, self.user_neighbors),
            axis=1,
            name='user_neighbor_reg')
        item_nbr_reg = tf.reduce_sum(
            tf.squared_difference(self.p_vectors, self.pos_neighbors),
            axis=1,
            name='pos_neighbor_reg') + tf.reduce_sum(
            tf.squared_difference(self.n_vectors, self.neg_neighbors),
            axis=1,
            name='neg_neighbor_reg'
        )
        return self.alpha_reg_nbr * tf.reduce_sum(user_nbr_reg + item_nbr_reg,
                                                  name='neigbor_reg')

    def _dist_regularization(self):
        dist = tf.reduce_sum(
            tf.math.squared_difference(self.u_vectors + self.pos_relations,
                                       self.p_vectors),
            axis=1)
        return self.alpha_reg_dist * tf.reduce_sum(dist,
                                                   name='distance_reg')

    def _create_inference(self):
        super(TransCF, self)._create_inference()
        self.pos_relations = self.pos_neighbors * self.user_neighbors
        self.neg_relations = self.neg_neighbors * self.user_neighbors

    def _pos_distances(self):
        distances = tf.reduce_sum(
            tf.math.squared_difference(self.u_vectors + self.pos_relations,
                                       self.p_vectors),
            axis=1,
            name='positive_distances'
        )
        return distances

    def _neg_distances(self):
        distances = tf.reduce_sum(
            tf.math.squared_difference(self.u_vectors + self.neg_relations,
                                       self.n_vectors),
            axis=1,
            name='negative_distances'
        )
        return distances

    def _build_eval_feedict(self, user_ids):
        uidx = [self.score_user_ids_dict[uid] for uid in user_ids]
        u_neighbors = self.dense_user_neighbors[uidx]
        ui_relations = np.expand_dims(u_neighbors, axis=1) * \
                       self.dense_item_neighbors
        return {
            self.score_user_ids: user_ids,
            self.scored_relations: ui_relations
        }

    def _create_score_items(self):
        """
        Calculate distance between scored users and all items
        :return:
        """
        self.logger.debug('--> Define TransCF ranking scores')
        u_vectors = tf.nn.embedding_lookup(self.user_embeddings,
                                           self.score_user_ids)
        # translate user vectors with relations
        translated_vectors = tf.expand_dims(u_vectors, axis=1) + \
                             self.scored_relations
        # get item vectors
        i_vectors = tf.expand_dims(self.item_embeddings, axis=0)

        # score = minus distance (N_USER, N_ITEM)
        self.scores = -tf.reduce_sum(
            tf.math.squared_difference(translated_vectors, i_vectors),
            axis=-1,
            name='scores')
