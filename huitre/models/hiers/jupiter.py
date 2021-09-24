import tensorflow as tf

from huitre.models.hiers.attram import AttRAM


class JUPITER(AttRAM):
    """
    Joint User Preference & Item Relations based Model
    """

    def get_relations_tf(self):
        return self.total_relations

    def _analyse_feed_dict(self, user_ids, item_ids):
        feed_dict = self._build_eval_feedict(user_ids)
        feed_dict[self.pos_ids] = item_ids
        return feed_dict

    def _create_inference(self):
        """
        Build inference graph
        :return:
        """
        super(JUPITER, self)._create_inference()
        self.logger.debug('--> Get JUPITER relations vectors')

        with tf.name_scope('inference'):
            self.pos_preferences = self._relations(tar_vectors=self.u_vectors,
                                                   ex_vectors=self.p_vectors,
                                                   key_matrix=self.key_matrix,
                                                   memories=self.memories)
            self.pos_relations = self._relations(tar_vectors=self.p_vectors,
                                                 ex_vectors=self.u_vectors,
                                                 key_matrix=self.key_matrix,
                                                 memories=self.memories,
                                                 catex=False)
            # translated user vectors
            self.total_relations = self.pos_preferences + self.pos_relations
            self.u_pos = self.u_vectors + self.pos_preferences + self.pos_relations
            if self.copy is False:
                self.logger.debug('----> Do not use positive relations '
                                  'for negative items')
                self.neg_preferences = self._relations(tar_vectors=self.u_vectors,
                                                       ex_vectors=self.n_vectors,
                                                       key_matrix=self.key_matrix,
                                                       memories=self.memories)
                self.neg_relations = self._relations(tar_vectors=self.n_vectors,
                                                     ex_vectors=self.u_vectors,
                                                     key_matrix=self.key_matrix,
                                                     memories=self.memories,
                                                     catex=False)
            else:
                self.neg_preferences = self.pos_preferences
                self.neg_relations = self.pos_relations
            self.u_neg = self.u_vectors + self.neg_preferences + self.neg_relations

    def _relations(self, tar_vectors, ex_vectors, key_matrix, memories, catex=True):
        # get item relations
        relations = self._relation_vectors(tar_vectors=tar_vectors,
                                           pref_vectors=self.pref_vectors,
                                           ex_vectors=ex_vectors,
                                           key_matrix=key_matrix,
                                           memories=memories,
                                           catex=catex)
        # get weighted relations
        att_relations = self._attentive_vectors(relations, ex_vectors)
        return att_relations

    def _pos_distances(self):
        self.logger.debug('--> Define JUPITER positive distances')
        distances = tf.reduce_sum(
            tf.squared_difference(self.u_pos, self.p_vectors),
            axis=1,
            name='pos_distances')
        return distances

    def _neg_distances(self):
        self.logger.debug('--> Define JUPITER negative distances')
        distances = tf.reduce_sum(
            tf.squared_difference(self.u_neg, self.n_vectors),
            axis=1,
            name='neg_distances')
        return distances

    def _create_score_items(self):
        u_vectors = tf.nn.embedding_lookup(self.user_embeddings,
                                           self.score_user_ids)
        # get user preferences vectors
        user_preferences = self._user_preference_vectors(u_vectors)

        # get item relations vectors
        item_relations = self._item_relations_vectors(u_vectors)

        # users = users + tf.expand_dims(u_vectors, axis=1)
        users = tf.expand_dims(u_vectors, axis=1) + user_preferences + \
                item_relations
        # scores = minus distance (N, M)
        self.scores = -tf.reduce_sum(
            tf.squared_difference(users, self.item_embeddings),
            axis=-1,
            name='scores')

    def _item_relations_vectors(self, u_vectors):
        pref_vectors = tf.transpose(self.pref_vectors, [0, 2, 1])

        item_relations = self._lram(tar_vectors=self.item_embeddings,
                                    ref_vectors=pref_vectors,
                                    key_matrix=self.key_matrix,
                                    memories=self.memories,
                                    is_scoring=True)
        # (n_users, n_items, pref_len, dim)
        embeddings = tf.transpose(item_relations, [0, 1, 3, 2]) * \
                     tf.expand_dims(
                         tf.expand_dims(u_vectors, axis=1), axis=-1)
        embeddings = tf.transpose(embeddings, [0, 1, 3, 2])
        relation_vectors = self._attentive_vectors_eval(embeddings, item_relations)
        return relation_vectors

    def _user_preference_vectors(self, u_vectors):
        # (n_users, n_items, dim)
        user_item_keys = tf.multiply(
            tf.expand_dims(u_vectors, axis=1),
            tf.expand_dims(self.item_embeddings, axis=0))
        # key_attention = tf.matmul(user_item_keys, self.fav_key_matrix)
        key_attention = tf.matmul(user_item_keys, self.key_matrix)
        key_attention = tf.nn.softmax(key_attention)
        # item_relations = tf.matmul(key_attention, self.fav_memories)
        item_relations = tf.matmul(key_attention, self.memories)
        # user preference vectors
        pref_relations = tf.gather(item_relations, self.pref_ids,
                                   axis=1, batch_dims=1)
        pref_relations = tf.expand_dims(pref_relations, axis=1)
        pref_relations = tf.tile(pref_relations,
                                 multiples=[1, self.n_items, 1, 1])
        # (n_users, n_items, pref_len+1, dim)
        relations = tf.concat(
            [tf.expand_dims(item_relations, axis=2), pref_relations],
            axis=2)
        embeddings = tf.transpose(relations, [0, 1, 3, 2]) * \
                     tf.expand_dims(
                         tf.expand_dims(self.item_embeddings, axis=0),
                         axis=-1)
        embeddings = tf.transpose(embeddings, [0, 1, 3, 2])
        res_vectors = self._attentive_vectors_eval(embeddings, relations)
        return res_vectors
