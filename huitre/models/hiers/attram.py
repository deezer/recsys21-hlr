from collections import defaultdict
import numpy as np
import tensorflow as tf

from huitre import HuitreError, RANDOM_PREF_SAMPLING, UNPOP_PREF_SAMPLING
from huitre.models.model import Model
from huitre.utils import add_mask


class AttRAM(Model):
    """
    Latent Relational Attentive Memory based model
    """
    def __init__(self, sess, params, n_users, n_items):
        super(AttRAM, self).__init__(sess, params, n_users, n_items)
        # number of slots in memory M
        self.num_memories = params['model']['params'].get(
            'num_memories', 10)
        # memory mode
        self.memory_mode = params['model']['params'].get(
            'memory_mode', 1)
        # copy relation to negative?
        self.copy = params['model']['params']['copy']
        self.beta = params['model']['params'].get('beta', 1.0)
        # user favorite items
        if 'train+valid' in params['ref_user_items']:
            self.pref_items = params['ref_user_items']['train+valid']
        else:
            self.pref_items = params['ref_user_items']['train']
        # for scalability reason, we should limit number of
        # interacted items because some users interact with
        # a couple of thousands of items
        self.max_num_prefs = params['model']['params'].get(
            'max_num_prefs', 20)
        self.clip_ram = params['model']['params'].get(
            'clip_ram', False)
        self.diff_ram = params['model']['params'].get(
            'diff_ram', False)
        pref_sampling = params.get('pref_sampling', 'random')
        self.pref_sampling = RANDOM_PREF_SAMPLING if \
            pref_sampling == 'random' else UNPOP_PREF_SAMPLING
        if self.pref_sampling != RANDOM_PREF_SAMPLING:
            self.logger.info('SAMPLING PREFS BASED ON ITEM RANKS')
            self.item_ranks = defaultdict(int)
            for _, iids in self.pref_items.items():
                for iid in iids:
                    self.item_ranks[iid] += 1

    def build_feedict(self, batch):
        feed_dict = super(AttRAM, self).build_feedict(batch)
        feed_dict[self.pref_ids] = batch[3]
        feed_dict[self.n_prefs] = batch[4]
        return feed_dict

    def _build_eval_feedict(self, user_ids):
        feed_dict = super(AttRAM, self)._build_eval_feedict(user_ids)
        feed_dict[self.user_ids] = user_ids
        pref_arr = []
        for uid in user_ids:
            prefs = list(self.pref_items[uid])
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
            pref_arr.append(prefs)
        num_prefs = np.array([len(pref) for pref in pref_arr])
        curr_max_num_prefs = min(self.max_num_prefs, np.max(num_prefs))
        pref_arr = add_mask(self.n_items, pref_arr, curr_max_num_prefs)
        feed_dict[self.pref_ids] = pref_arr
        feed_dict[self.n_prefs] = num_prefs
        return feed_dict

    def _create_placeholders(self):
        super(AttRAM, self)._create_placeholders()
        self.logger.debug('--> Create additional AttRAM placeholders')
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
        super(AttRAM, self)._create_variables()
        self.logger.debug('--> Create additional MEM variables for ALRM')
        with tf.name_scope('lram'):
            self.key_matrix = tf.compat.v1.get_variable(
                name='key_matrix',
                shape=[self.embedding_dim, self.num_memories],
                initializer=tf.random_normal_initializer(
                    0., stddev=1. / (self.embedding_dim ** 0.5)),
                dtype=tf.float32)
            self.memories = tf.compat.v1.get_variable(
                name='memory',
                shape=[self.num_memories, self.embedding_dim],
                initializer=tf.random_normal_initializer(
                    0., stddev=1. / (self.embedding_dim ** 0.5)),
                dtype=tf.float32)
            if self.diff_ram is True:
                self.item_key_matrix = tf.compat.v1.get_variable(
                    name='item_key_matrix',
                    shape=[self.embedding_dim, self.num_memories],
                    initializer=tf.random_normal_initializer(
                        0., stddev=1. / (self.embedding_dim ** 0.5)),
                    dtype=tf.float32)
                self.item_memories = tf.compat.v1.get_variable(
                    name='item_memory',
                    shape=[self.num_memories, self.embedding_dim],
                    initializer=tf.random_normal_initializer(
                        0., stddev=1. / (self.embedding_dim ** 0.5)),
                    dtype=tf.float32)
        with tf.name_scope('user_item_embeddings'):
            # use to mask items which are not interacted with user
            zero_item = tf.constant(
                0.0, tf.float32, [1, self.embedding_dim],
                name='zero_item')
            self.ctx_item_embeddings = tf.concat(
                [self.item_embeddings, zero_item],
                axis=0,
                name='ctx_item_embeddings')

    def _lram(self, tar_vectors, ref_vectors, key_matrix, memories,
              is_scoring=False):
        """
        Generate relation given user vectors and item vectors
        :param tar_vectors:
        :param ref_vectors:
        :param is_scoring:
        :return:
        """
        # get relation key by Hadamard product
        key_att = self._key_attention(tar_vectors, ref_vectors, key_matrix,
                                      is_scoring)
        key_att = tf.nn.softmax(key_att)
        if self.memory_mode == 1:
            relation = tf.matmul(key_att, memories)
        else:
            raise HuitreError(f'Not support memory mode {self.memory_mode}!'
                              f'Only `1` for the moment')
        return relation

    def _relation_vectors(self, tar_vectors, pref_vectors,
                          ex_vectors, key_matrix, memories,
                          is_scoring=False, catex=True):
        """
        Calculate relation vectors
        :param tar_vectors:
        :param pref_vectors:
        :param ex_vectors:
        :param key_matrix:
        :param memories:
        :param is_scoring:
        :return:
        """
        if catex is True:
            pref_vectors = tf.transpose(pref_vectors, [1, 0, 2])
            ex_vectors = tf.expand_dims(ex_vectors, axis=0)
            pref_vectors = tf.concat([ex_vectors, pref_vectors],
                                     axis=0)
            pref_vectors = tf.transpose(pref_vectors, [1, 2, 0])
        else:
            self.logger.debug('----> NOT concatenate ex_vectors to pref_vectors')
            pref_vectors = tf.transpose(self.pref_vectors, [0, 2, 1])
        relations = self._lram(tar_vectors=tar_vectors,
                               ref_vectors=pref_vectors,
                               key_matrix=key_matrix,
                               memories=memories,
                               is_scoring=is_scoring)
        return relations

    @classmethod
    def _key_attention(cls, tar_vectors, ref_vectors, key_matrix,
                       is_scoring=False):
        """
        Hadamard between tar_vectors and ref_vectors
        :param tar_vectors:
        :param ref_vectors:
        :param key_matrix:
        :param is_scoring:
        :return:
        """
        if is_scoring is not True:
            if tar_vectors.shape.as_list() == ref_vectors.shape.as_list():
                tar_ref_keys = tar_vectors * ref_vectors
                key_att = tf.matmul(tar_ref_keys, key_matrix)
            else:
                tar_vectors = tf.expand_dims(tar_vectors, axis=-1)
                tar_ref_keys = tar_vectors * ref_vectors
                key_att = tf.matmul(
                    tf.transpose(tar_ref_keys, [0, 2, 1]),
                    key_matrix)
        else:
            if len(tar_vectors.shape.as_list()) != len(ref_vectors.shape.as_list()):
                tar_ref_keys = tf.expand_dims(tar_vectors, axis=-1) * \
                               tf.expand_dims(ref_vectors, axis=1)
                key_att = tf.matmul(
                    tf.transpose(tar_ref_keys, [0, 1, 3, 2]),
                    key_matrix)
            else:
                tar_ref_keys = tf.multiply(
                    tf.expand_dims(tar_vectors, axis=1),
                    tf.expand_dims(ref_vectors, axis=0))
                key_att = tf.matmul(tar_ref_keys, key_matrix)
        return key_att

    def _create_inference(self):
        """
        Build inference graph
        :return:
        """
        super(AttRAM, self)._create_inference()
        with tf.name_scope('inference'):
            # preference item vectors [batch_size, dim, n_prefs]
            self.pref_vectors = tf.nn.embedding_lookup(
                self.ctx_item_embeddings,
                self.pref_ids)

    def _attentive_vectors(self, relations, ex_vectors):
        with tf.name_scope('attention_network'):
            embeddings = relations * tf.expand_dims(
                ex_vectors, axis=1)
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
                self.logger.debug(f'----> BETA ATTENTION: {self.beta}')
                exp_sum = tf.pow(exp_sum,
                                 tf.constant(self.beta, tf.float32, [1]))
            att = tf.div(exp_wij, exp_sum)
            att_vecs = tf.reduce_sum(
                relations * tf.expand_dims(att, axis=-1), axis=1)
        return att_vecs

    def _attentive_vectors_eval(self, embeddings, relations):
        with tf.name_scope('attention_network_eval'):
            # (n_users, n_items, pref_len+1)
            w_ij = tf.reduce_sum(embeddings, axis=-1)
            exp_wij = tf.exp(w_ij)
            n_pr = tf.shape(embeddings)[2]
            n_prefs = tf.add(self.n_prefs, 1.0)
            mask_mat = tf.sequence_mask(n_prefs, maxlen=n_pr,
                                        dtype=tf.float32)
            # (n_users, pref_len+1, n_items)
            exp_wij = tf.expand_dims(mask_mat, axis=-1) * tf.transpose(
                exp_wij, [0, 2, 1])
            exp_sum = tf.reduce_sum(exp_wij, axis=1)
            if self.beta != 1.0:
                self.logger.debug(f'----> BETA ATTENTION: {self.beta}')
                exp_sum = tf.pow(exp_sum,
                                 tf.constant(self.beta, tf.float32, [1]))
            # (n_users, pref_len+1, n_items)
            att = exp_wij / tf.expand_dims(exp_sum, axis=1)
            att_vecs = tf.expand_dims(att, axis=-1) * \
                       tf.transpose(relations, [0, 2, 1, 3])
            # (n_users, n_items, dim)
            att_vecs = tf.reduce_sum(att_vecs, axis=1)
        return att_vecs

    def _clip_by_norm_op(self):
        ops = super(AttRAM, self)._clip_by_norm_op()
        if self.clip_ram is not True:
            return ops
        self.logger.debug('--> AttRAM with CLIPRAM')
        ops = ops + [
            tf.compat.v1.assign(self.key_matrix,
                                tf.clip_by_norm(self.key_matrix,
                                                self.clip_norm,
                                                axes=[1])),
            tf.compat.v1.assign(self.memories,
                                tf.clip_by_norm(self.memories,
                                                self.clip_norm,
                                                axes=[1]))
        ]
        if self.diff_ram is True:
            self.logger.debug('--> CLIPRAM & DIFFRAM')
            ops = ops + ops + [
                tf.compat.v1.assign(self.item_key_matrix,
                                    tf.clip_by_norm(self.item_key_matrix,
                                                    self.clip_norm,
                                                    axes=[1])),
                tf.compat.v1.assign(self.item_memories,
                                    tf.clip_by_norm(self.item_memories,
                                                    self.clip_norm,
                                                    axes=[1]))
            ]
        return ops

    def _pos_distances(self):
        raise NotImplementedError('_pos_distances method should be '
                                  'implemented in concrete model')

    def _neg_distances(self):
        raise NotImplementedError('_neg_distances method should be '
                                  'implemented in concrete model')

    def _create_score_items(self):
        raise NotImplementedError('_create_score_items method should be '
                                  'implemented in concrete model')
