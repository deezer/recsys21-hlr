import tensorflow as tf

from huitre.models.hiers.attram import AttRAM


class HLRE(AttRAM):
    """
    Hierarchical Latent Relations with TransE
    """
    EPS = 1e-8

    def get_relations_tf(self):
        return self.pos_relations

    def _analyse_feed_dict(self, user_ids, item_ids):
        feed_dict = self._build_eval_feedict(user_ids)
        feed_dict[self.pos_ids] = item_ids
        return feed_dict

    def _create_inference(self):
        """
        Build inference graph
        :return:
        """
        super(HLRE, self)._create_inference()
        # (batch_size, pref_len+1, dim)
        catex = not self.diff_ram
        pos_relations = self._relation_vectors(tar_vectors=self.p_vectors,
                                               pref_vectors=self.pref_vectors,
                                               ex_vectors=self.u_vectors,
                                               key_matrix=self.key_matrix,
                                               memories=self.memories,
                                               catex=catex)
        if self.diff_ram is True:
            ui_relations = self._lram(tar_vectors=self.u_vectors,
                                      ref_vectors=self.p_vectors,
                                      key_matrix=self.item_key_matrix,
                                      memories=self.item_memories)
            pos_relations = tf.concat(
                [tf.expand_dims(ui_relations, axis=1), pos_relations],
                axis=1)
        self.pos_relations = self._attentive_vectors(relations=pos_relations,
                                                     ex_vectors=self.u_vectors)
        if self.copy is False:
            self.logger.debug('----> Do not use positive relations '
                              'for negative items')
            neg_relations = self._relation_vectors(tar_vectors=self.n_vectors,
                                                   pref_vectors=self.pref_vectors,
                                                   ex_vectors=self.u_vectors,
                                                   key_matrix=self.key_matrix,
                                                   memories=self.memories)
            self.neg_relations = self._attentive_vectors(relations=neg_relations,
                                                         ex_vectors=self.u_vectors)
        else:
            self.logger.debug('----> COPY positive relations to negative relations')
            self.neg_relations = self.pos_relations

    def _pos_distances(self):
        self.logger.debug('--> Define HLRE positive distances')
        distances = tf.reduce_sum(
            tf.squared_difference(self.u_vectors + self.pos_relations,
                                  self.p_vectors),
            axis=1,
            name='pos_distances')
        return distances

    def _neg_distances(self):
        self.logger.debug('--> Define HLRE negative distances')
        distances = tf.reduce_sum(
            tf.squared_difference(self.u_vectors + self.neg_relations,
                                  self.n_vectors),
            axis=1,
            name='neg_distances')
        return distances

    def _create_score_items(self):
        self.logger.debug('--> Define HLRE ranking scores')
        u_vectors = tf.nn.embedding_lookup(self.user_embeddings,
                                           self.score_user_ids)
        # (n_users, n_items, n_prefs, dim)
        catex = not self.diff_ram
        item_relations = self._relation_vectors(tar_vectors=self.item_embeddings,
                                                pref_vectors=self.pref_vectors,
                                                ex_vectors=u_vectors,
                                                key_matrix=self.key_matrix,
                                                memories=self.memories,
                                                catex=catex,
                                                is_scoring=True)
        if self.diff_ram is True:
            ui_relations = self._lram(tar_vectors=u_vectors,
                                      ref_vectors=self.item_embeddings,
                                      key_matrix=self.item_key_matrix,
                                      memories=self.item_memories,
                                      is_scoring=True)
            item_relations = tf.concat(
                [tf.expand_dims(ui_relations, axis=2), item_relations],
                axis=2)
        # (n_users, n_items, pref_len, dim)
        embeddings = tf.transpose(item_relations, [0, 1, 3, 2]) * \
                     tf.expand_dims(
                         tf.expand_dims(u_vectors, axis=1), axis=-1)
        embeddings = tf.transpose(embeddings, [0, 1, 3, 2])
        # weighted relations
        relations = self._attentive_vectors_eval(embeddings,
                                                 item_relations)
        users = tf.expand_dims(u_vectors, axis=1) + relations
        # scores = minus distance (N, M)
        self.scores = -tf.reduce_sum(
            tf.squared_difference(users, self.item_embeddings),
            axis=-1,
            name='scores')
