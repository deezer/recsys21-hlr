import tensorflow as tf

from huitre.models.model import Model


class CML(Model):
    """
    Collaborative Metric Learning
    :reference: Hsieh, C. K. et al. Collaborative metric learning.
                In Proceedings WWW 2017.
    """
    def _pos_distances(self):
        """
        Get distance from user to positive item
        :return:
        """
        distances = tf.reduce_sum(
            tf.math.squared_difference(self.u_vectors, self.p_vectors),
            axis=1,
            name='positive_distances'
        )
        return distances

    def _neg_distances(self):
        """
        Get distance from user to negative item
        :return:
        """
        distances = tf.reduce_sum(
            tf.math.squared_difference(self.u_vectors, self.n_vectors),
            axis=1,
            name='negative_distances'
        )
        return distances

    def _create_score_items(self):
        """
        Calculate distance between scored users and all items
        :return:
        """
        self.logger.debug('--> Define triplet ranking scores')
        # get user vectors to score
        # (N_USER_IDS, 1, K)
        u_vectors = tf.expand_dims(
            tf.nn.embedding_lookup(self.user_embeddings,
                                   self.score_user_ids),
            axis=1)
        # get item vectors
        # (1, N_ITEM, K)
        i_vectors = tf.expand_dims(self.item_embeddings, axis=0)

        # score = minus distance (N_USER, N_ITEM)
        self.scores = -tf.reduce_sum(
            tf.math.squared_difference(u_vectors, i_vectors),
            axis=2,
            name='scores')
