import tensorflow as tf

from huitre.models.model import Model


class BPR(Model):
    """
    Bayesian Personalized Ranking for implicit feedback [1]
    References
    ----------
    [1]. Rendle, S. et al., 2009, BPR: Bayesian Personalized Ranking from Implicit Feedback,
         Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI2009)
    """

    def __init__(self, sess, params, n_users, n_items):
        super(BPR, self).__init__(sess, params, n_users, n_items)
        # regularization for user factors
        self.user_regularization = getattr(params,
                                           'user_regularization',
                                           0.0025)
        # regularization for user bias
        self.user_bias_regularization = getattr(params,
                                                'user_bias_regularization',
                                                0.001)

        # regularization for item factors
        self.pos_item_regularization = getattr(params,
                                               'pos_item_regularization',
                                               0.0025)
        self.neg_item_regularization = getattr(params,
                                               'neg_item_regularization',
                                               0.00025)

        # regularization for item biases
        self.item_bias_regularization = getattr(params,
                                                'item_bias_regularization',
                                                0.001)

    def _create_variables(self):
        super(BPR, self)._create_variables()
        # user bias
        self.user_bias_mat = tf.get_variable(name='user_bias',
                                             shape=[self.n_users, 1],
                                             initializer=tf.constant_initializer(0.0))
        # item bias
        self.item_bias = tf.get_variable(name='item_bias',
                                         shape=[self.n_items, 1],
                                         initializer=tf.constant_initializer(0.0))

    def _create_inference(self):
        super(BPR, self)._create_inference()
        self.user_bias = tf.nn.embedding_lookup(self.user_bias_mat,
                                                self.user_ids)
        self.pos_item_bias = tf.nn.embedding_lookup(self.item_bias,
                                                    self.pos_ids)
        self.neg_item_bias = tf.nn.embedding_lookup(self.item_bias,
                                                    self.neg_ids)

    def _pos_distances(self):
        return self.pos_item_bias + tf.reduce_sum(tf.multiply(self.u_vectors,
                                                              self.p_vectors),
                                                  axis=1,
                                                  keep_dims=True)

    def _neg_distances(self):
        return self.neg_item_bias + tf.reduce_sum(tf.multiply(self.u_vectors,
                                                              self.n_vectors),
                                                  axis=1,
                                                  keep_dims=True)

    def l2_norm(self):
        self.logger.debug('DEFINE BPR L2 REGULARIZATION')
        norm = tf.add_n([
            self.user_regularization * tf.reduce_sum(tf.multiply(self.u_vectors, self.u_vectors)),
            self.user_bias_regularization * tf.reduce_sum(
                tf.multiply(self.user_bias, self.user_bias)),
            self.pos_item_regularization * tf.reduce_sum(
                tf.multiply(self.p_vectors, self.p_vectors)),
            self.neg_item_regularization * tf.reduce_sum(
                tf.multiply(self.n_vectors, self.n_vectors)),
            self.item_bias_regularization * tf.reduce_sum(
                tf.multiply(self.pos_item_bias, self.pos_item_bias)),
            self.item_bias_regularization * tf.reduce_sum(
                tf.multiply(self.neg_item_bias, self.neg_item_bias))
        ])
        return norm

    def _create_loss(self):
        """
        Build loss graph
        :return:
        """
        self.logger.debug('--> Define BPR loss')
        # positive distances
        self.pos_distances = self._pos_distances()
        # negative distances
        self.neg_distances = self._neg_distances()
        self.x_hat = self.pos_distances - self.neg_distances
        self.loss = self.l2_norm() - tf.reduce_sum(tf.log(tf.sigmoid(self.x_hat)))

    def _create_train_ops(self):
        """
        Train operations
        :return:
        """
        self.logger.debug('--> Define training operators')
        optimizer = self._build_optimizer(self.learning_rate)
        self.train_ops = [optimizer.minimize(self.loss)]

    def _create_score_items(self):
        """
        Calculate distance between scored users and all items
        :return:
        """
        self.logger.debug('--> Define BPR ranking scores')
        # get user vectors to score
        # (N_USER_IDS, 1, K)
        u_vectors = tf.expand_dims(
            tf.nn.embedding_lookup(self.user_embeddings,
                                   self.score_user_ids),
            axis=1)
        # get item vectors
        # (1, N_ITEM, K)
        i_vectors = tf.expand_dims(self.item_embeddings, axis=0)
        item_bias = tf.transpose(self.item_bias)
        self.scores = item_bias + tf.reduce_sum(tf.multiply(u_vectors, i_vectors), 2,
                                                name='scores')
