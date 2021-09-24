import numpy as np
import tensorflow as tf

from huitre.models.model import Model
from huitre.utils import add_mask


class AttCML(Model):
    """
    Attentive CML
    """
    def __init__(self, sess, params, n_users, n_items):
        super(AttCML, self).__init__(sess, params, n_users, n_items)
        self.beta = params['model']['params'].get('beta', 1.0)
        self.copy = params['model']['params'].get('copy', False)
        if 'train+valid' in params['ref_user_items']:
            self.pref_items = params['ref_user_items']['train+valid']
        else:
            self.pref_items = params['ref_user_items']['train']
        self.max_num_prefs = params['model']['params'].get(
            'max_num_prefs', 20)

    def build_feedict(self, batch):
        feed_dict = super(AttCML, self).build_feedict(batch)
        feed_dict[self.pref_ids] = batch[3]
        feed_dict[self.n_prefs] = batch[4]
        return feed_dict

    def _create_placeholders(self):
        super(AttCML, self)._create_placeholders()
        self.logger.debug('--> Create additional AttCML placeholders')
        with tf.name_scope('input_data'):
            # item_ids in user preference
            self.pref_ids = tf.compat.v1.placeholder(
                name='pref_ids', dtype=tf.int32,
                shape=[None, None])
            # number of preferenced item ids: size = (batch, 1)
            self.n_prefs = tf.compat.v1.placeholder(
                name='num_preference_items',
                dtype=tf.float32,
                shape=[None])

    def _create_variables(self):
        """
        Build variables
        :return:
        """
        super(AttCML, self)._create_variables()
        self.logger.debug('--> Create context Item embeddings in AttCML')
        with tf.name_scope('user_item_embeddings'):
            # use to mask items which are not interacted with user
            zero_item = tf.constant(
                0.0, tf.float32, [1, self.embedding_dim],
                name='zero_item')
            self.ctx_item_embeddings = tf.concat(
                [self.item_embeddings, zero_item],
                axis=0,
                name='ctx_item_embeddings')

    def _create_inference(self):
        """
        Build inference graph
        :return:
        """
        super(AttCML, self)._create_inference()
        self.logger.debug('--> Get preference vectors for a batch')
        with tf.name_scope('inference'):
            # preference item vectors [batch_size, n_prefs, dim]
            self.pref_vectors = tf.nn.embedding_lookup(self.ctx_item_embeddings,
                                                       self.pref_ids)
            # translated user vectors
            self.u_pos_att = self._attentive_vectors(
                self.pref_vectors * tf.expand_dims(self.p_vectors, axis=1))
            self.u_pos = self.u_vectors + self.u_pos_att

            if self.copy is not True:
                self.u_neg_att = self._attentive_vectors(
                    self.pref_vectors * tf.expand_dims(self.n_vectors, axis=1))
            else:
                self.u_neg_att = self.u_pos_att
            self.u_neg = self.u_vectors + self.u_neg_att

    def _attentive_vectors(self, embeddings):
        with tf.name_scope('attention_network'):
            w_ij = tf.reduce_sum(embeddings, axis=-1)
            exp_wij = tf.exp(w_ij)
            n_pr = tf.shape(embeddings)[1]
            n_prefs = tf.add(self.n_prefs, 1.0)
            mask_mat = tf.sequence_mask(n_prefs,
                                        maxlen=n_pr,
                                        dtype=tf.float32)
            exp_wij = mask_mat * exp_wij
            exp_sum = tf.reduce_sum(exp_wij, axis=1, keep_dims=True)
            if self.beta != 1.0:
                exp_sum = tf.pow(exp_sum,
                                 tf.constant(self.beta, tf.float32, [1]))
            att = tf.div(exp_wij, exp_sum)
            att_vecs = tf.reduce_sum(
                self.pref_vectors * tf.expand_dims(att, axis=-1), axis=1)
        return att_vecs

    def _attentive_vectors_eval(self, embeddings):
        with tf.name_scope('attention_network_eval'):
            w_ij = tf.reduce_sum(embeddings, axis=-1)
            exp_wij = tf.exp(w_ij)
            n_pr = tf.shape(embeddings)[2]
            # n_prefs = tf.add(self.n_prefs, 1.0)
            mask_mat = tf.sequence_mask(self.n_prefs, maxlen=n_pr,
                                        dtype=tf.float32)
            exp_wij = tf.expand_dims(mask_mat, axis=-1) * tf.transpose(
                exp_wij, [0, 2, 1])
            exp_sum = tf.reduce_sum(exp_wij, axis=1)
            if self.beta != 1.0:
                exp_sum = tf.pow(exp_sum,
                                 tf.constant(self.beta, tf.float32, [1]))
            att = exp_wij / tf.expand_dims(exp_sum, axis=1)
            att_vecs = tf.expand_dims(att, axis=-1) * \
                       tf.expand_dims(self.pref_vectors, axis=2)
            att_vecs = tf.reduce_sum(att_vecs, axis=1)
        return att_vecs

    def _pos_distances(self):
        self.logger.debug('--> Define AttCML positive distances')
        distances = tf.reduce_sum(
            tf.squared_difference(self.u_pos, self.p_vectors),
            axis=1,
            name='pos_distances')
        return distances

    def _neg_distances(self):
        self.logger.debug('--> Define AttCML negative distances')
        distances = tf.reduce_sum(
            tf.squared_difference(self.u_neg, self.n_vectors),
            axis=1,
            name='neg_distances')
        return distances

    def _create_score_items(self):
        u_vectors = tf.nn.embedding_lookup(self.user_embeddings,
                                           self.score_user_ids)
        # (n_users, n_items, pref_len, dim)
        item_dot_prefs = tf.expand_dims(self.pref_vectors, axis=1) * \
                         tf.expand_dims(self.item_embeddings, axis=1)
        users = self._attentive_vectors_eval(item_dot_prefs)
        users = users[:, :self.n_items, :]
        users = users + tf.expand_dims(u_vectors, axis=1)
        # scores = minus distance (N, M)
        self.scores = -tf.reduce_sum(
            tf.squared_difference(users, self.item_embeddings),
            axis=-1,
            name='scores')

    def _build_eval_feedict(self, user_ids):
        feed_dict = super(AttCML, self)._build_eval_feedict(user_ids)
        pref_arr = []
        for uid in user_ids:
            prefs = list(self.pref_items[uid])
            if len(prefs) > self.max_num_prefs:
                prefs = np.random.choice(prefs,
                                         size=self.max_num_prefs,
                                         replace=False)
            pref_arr.append(prefs)
        num_prefs = np.array([len(pref) for pref in pref_arr])
        curr_max_num_prefs = min(self.max_num_prefs, np.max(num_prefs))
        pref_arr = add_mask(self.n_items, pref_arr, curr_max_num_prefs)
        feed_dict[self.pref_ids] = pref_arr
        feed_dict[self.n_prefs] = num_prefs
        return feed_dict
