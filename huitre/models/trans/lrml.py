import tensorflow as tf

from huitre.models.model import Model


class LRML(Model):
    """
    Latent Relational Metric Learning
    :reference Tay Y. et al. "Latent relational metric learning via memory-based
    attention for collaborative ranking." Proceedings of WWW 2018.
    """

    def __init__(self, sess, params, n_users, n_items):
        """
        Initialize model
        :param sess:
        :param params:
        :param n_users:
        :param n_items:
        :return
        """
        super(LRML, self).__init__(sess, params, n_users, n_items)
        # number of memories used in LRAM
        self.num_memories = params['model']['params'].get(
            'num_memories', 10)
        # copy relation to negative?
        self.copy = params['model']['params']['copy']
        self.clip_ram = params['model']['params'].get('clip_ram', False)

    def get_relations_tf(self):
        return self.pos_relations

    def _analyse_feed_dict(self, user_ids, item_ids):
        feed_dict = {
            self.user_ids: user_ids,
            self.pos_ids: item_ids
        }
        return feed_dict

    def _create_variables(self):
        super(LRML, self)._create_variables()
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

    def _lram(self, user_vectors, item_vectors, memories, is_scoring=False):
        """
        Generate relation given user vectors and item vectors
        :param user_vectors:
        :param item_vectors:
        :param memories:
        :param is_scoring:
        :return:
        """
        # get relation key by Hadamard product
        if is_scoring is not True:
            user_item_keys = user_vectors * item_vectors
            key_attention = tf.matmul(user_item_keys,
                                      self.key_matrix)
        else:
            user_item_keys = tf.multiply(
                tf.expand_dims(user_vectors, axis=1),
                tf.expand_dims(item_vectors, axis=0))
            key_attention = tf.matmul(user_item_keys,
                                      self.key_matrix)
        key_attention = tf.nn.softmax(key_attention,
                                      axis=-1)
        relation = tf.matmul(key_attention, memories)
        return relation

    def _create_inference(self):
        super(LRML, self)._create_inference()
        self.pos_relations = self._lram(
            user_vectors=self.u_vectors,
            item_vectors=self.p_vectors,
            memories=self.memories)
        if self.copy is False:
            self.logger.debug('----> Do not use positive relations '
                              'for negative items')
            self.neg_relations = self._lram(user_vectors=self.u_vectors,
                                            item_vectors=self.n_vectors,
                                            memories=self.memories)
        else:
            self.logger.debug('----> COPY positive relations '
                              'for negative items')
            self.neg_relations = self.pos_relations

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

    def _clip_by_norm_op(self):
        ops = super(LRML, self)._clip_by_norm_op()
        if self.clip_ram is not True:
            return ops
        self.logger.debug('--> LRML with CLIPRAM')
        return ops + [
            tf.compat.v1.assign(self.key_matrix,
                                tf.clip_by_norm(self.key_matrix,
                                                self.clip_norm,
                                                axes=[1])),
            tf.compat.v1.assign(self.memories,
                                tf.clip_by_norm(self.memories,
                                                self.clip_norm,
                                                axes=[1]))
        ]

    def _create_score_items(self):
        """
        Calculate distance between scored users and all items
        :return:
        """
        self.logger.debug('--> Define LRML ranking scores')
        u_vectors = tf.nn.embedding_lookup(self.user_embeddings,
                                           self.score_user_ids)
        # get user/item relations
        relations = self._lram(u_vectors,
                               self.item_embeddings,
                               memories=self.memories,
                               is_scoring=True)
        # translate user vectors with relations
        translated_vectors = tf.expand_dims(u_vectors, axis=1) + relations
        # get item vectors
        i_vectors = tf.expand_dims(self.item_embeddings, axis=0)

        # score = minus distance (N_USER, N_ITEM)
        self.scores = -tf.sqrt(tf.reduce_sum(
            tf.math.squared_difference(translated_vectors, i_vectors),
            axis=-1,
            name='scores'))
