import os
import pickle
import toolz
import numpy as np
import tensorflow as tf

from huitre import HuitreError
from huitre.logging import get_logger
from huitre import RANDOM_PREF_SAMPLING, UNPOP_PREF_SAMPLING
from huitre.utils import best_cosine_similarities


class Model:
    """
    Abstract model
    """
    # Supported optimizers.
    ADADELTA = 'Adadelta'
    SGD = 'SGD'
    ADAM = 'Adam'

    def __init__(self, sess, params, n_users, n_items):
        """
        Initialize a model
        :param sess: global session
        :param params: model parameters
        :param n_users: number of users
        :param n_items: number of items
        """
        self.logger = get_logger()
        self.sess = sess
        self.learning_rate = params.get('learning_rate', 0.001)
        self.embedding_dim = params.get('embedding_dim', 32)
        self.model_dir = params.get('model_dir', 'exp/model')
        self.n_epochs = params.get('n_epochs', 20)
        self.n_negatives = params['model']['params']['n_negatives']
        self.clip_norm = params['model']['params'].get('clip_norm', 1.0)
        self.margin = params['model']['params'].get('margin', 1.0)
        self.n_users = n_users
        self.n_items = n_items
        self.optimizer = params['optimizer']
        # loss
        self.loss = None
        self.reg_loss = None
        # recommendation scores
        self.scores = None
        # model saving
        self.checkpoint = None
        self.saver = None
        self.max_to_keep = params.get('max_to_keep', 1)
        self.pref_sampling = params.get('pref_sampling', 'random')
        self.pref_sampling = RANDOM_PREF_SAMPLING \
            if self.pref_sampling == 'random' else UNPOP_PREF_SAMPLING

    def build_graph(self):
        """
        Build model computation graph
        :return:
        """
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_score_items()
        self._create_train_ops()
        self.saver = tf.compat.v1.train.Saver(
            max_to_keep=self.max_to_keep)

    def recommend(self, users, excluded_items, num_items,
                  n_users_in_chunk=100):
        """
        Recommend num_items of items to users
        :param users: user_ids array
        :param excluded_items: dict(set) excluded items for each users
        :param num_items:
        :param n_users_in_chunk:
        :return:
        """
        max_train_pref_len = np.max(
            [len(prefs) for _, prefs in excluded_items.items()])
        self.logger.debug(f'Max preference len in train: {max_train_pref_len}')
        reco_items = {}
        for user_ids in toolz.partition_all(n_users_in_chunk, users):
            feed_dict = self._build_eval_feedict(user_ids)
            reco_items_chunk = self._topk_items(feed_dict,
                                                k=num_items + max_train_pref_len)
            for uid, reco_iids in reco_items_chunk.items():
                filtered_iids = [iid for iid in reco_iids
                                 if iid not in excluded_items[uid]]
                reco_items[uid] = filtered_iids[:num_items]
        return reco_items

    def embeddings(self):
        user_embeddings, item_embeddings = self.sess.run(
            [self.user_embeddings, self.item_embeddings])
        return user_embeddings, item_embeddings

    def relation_vectors(self):
        pass

    def save(self, save_path, global_step):
        """
        Save the model to directory
        :param save_path:
        :param global_step:
        :return:
        """
        self.saver.save(self.sess, save_path=save_path,
                        global_step=global_step)

    def restore(self):
        """
        Restore the model if it already exists
        :return:
        """
        self.checkpoint = tf.compat.v1.train.get_checkpoint_state(
            self.model_dir)
        if self.checkpoint is not None:
            self.logger.info(f'Load {self.__class__} model from {self.model_dir}')
            self.build_graph()
            self.saver.restore(self.sess, self.checkpoint.model_checkpoint_path)

    def analyse(self, inputs):
        similarities_path = inputs['cosine_similarities_path']
        if not os.path.exists(similarities_path):
            self.logger.debug('Load relations and corresponding user_ids, item_ids')
            user_ids, item_ids, relations = self._get_relations(inputs, self.get_relations_tf())
            self.logger.debug(f'Number of relations: {len(relations)}')
            self.logger.debug(f'Extract only {inputs["num_interactions"]} '
                              f'relations to analyse')
            rng = np.random.RandomState(inputs['random_state'])
            indexes = np.arange(len(relations))
            rng.shuffle(indexes)
            indexes = indexes[:inputs['num_interactions']]
            user_ids = user_ids[indexes]
            item_ids = item_ids[indexes]
            relations = relations[indexes]
            self.logger.debug('Calculate best relations cosine similarities')
            best_cosim_relation_indexes = best_cosine_similarities(relations)
            best_relations = {
                'user_ids': user_ids,
                'item_ids': item_ids,
                'relation_indexes': best_cosim_relation_indexes
            }
            pickle.dump(best_relations,
                        open(similarities_path, 'wb'))
        else:
            self.logger.debug('Load best relations cosine similarities')
            best_relations = pickle.load(open(similarities_path, 'rb'))
        # get matching probability
        self._meta_match(user_ids=best_relations['user_ids'],
                         item_ids=best_relations['item_ids'],
                         user_meta=inputs['user_meta'],
                         item_meta=inputs['item_meta'],
                         best_relation_indexes=best_relations['relation_indexes'])

    def build_feedict(self, batch):
        feed_dict = {
            self.user_ids: batch[0],
            self.pos_ids: batch[1],
            self.neg_ids: batch[2].flatten()
        }
        return feed_dict

    def _create_placeholders(self):
        """
        Build input graph
        :return:
        """
        self.logger.debug('--> Create triplet placeholders')
        with tf.name_scope('input_data'):
            # batch of user ids
            self.user_ids = tf.compat.v1.placeholder(name='user_ids',
                                                     dtype=tf.int32,
                                                     shape=[None])
            # batch of positive item ids (items have interactions with
            # corresponding users presented in user_ids)
            self.pos_ids = tf.compat.v1.placeholder(name='pos_ids',
                                                    dtype=tf.int32,
                                                    shape=[None])
            # batch of negative item ids (items do not have
            # interactions with users presented in user_ids)
            self.neg_ids = tf.compat.v1.placeholder(name='neg_ids',
                                                    dtype=tf.int32,
                                                    shape=[None])
            # user ids used to score model
            self.score_user_ids = tf.compat.v1.placeholder(name='scored_user_ids',
                                                           dtype=tf.int32,
                                                           shape=[None])

    def _create_variables(self):
        """
        Build variables
        :return:
        """
        self.logger.debug('--> Create User/Item embeddings')
        with tf.name_scope('user_item_embeddings'):
            # user embeddings
            self.user_embeddings = tf.compat.v1.get_variable(
                name='user_embedding_matrix',
                shape=[self.n_users, self.embedding_dim],
                initializer=tf.random_normal_initializer(
                    0., stddev=1. / (self.embedding_dim ** 0.5)),
                dtype=tf.float32
            )
            # item embeddings
            self.item_embeddings = tf.compat.v1.get_variable(
                name='item_embedding_matrix',
                shape=[self.n_items, self.embedding_dim],
                initializer=tf.random_normal_initializer(
                    0., stddev=1. / (self.embedding_dim ** 0.5)),
                dtype=tf.float32
            )

    def _create_inference(self):
        """
        Build inference graph
        :return:
        """
        self.logger.debug('--> Get user/pos_items/neg_items vectors '
                          'for a batch')
        with tf.name_scope('inference'):
            # user vectors [batch_size, dim]
            self.u_vectors = tf.nn.embedding_lookup(self.user_embeddings,
                                                    self.user_ids,
                                                    name='batch_user_vectors')
            # positive item vectors [batch_size, dim]
            self.p_vectors = tf.nn.embedding_lookup(self.item_embeddings,
                                                    self.pos_ids,
                                                    name='batch_positive_vectors')
            # negative item vectors [batch_size, dim, n_negs]
            self.n_vectors = tf.nn.embedding_lookup(self.item_embeddings,
                                                    self.neg_ids,
                                                    name='batch_negative_vectors')

    def _create_loss(self):
        """
        Build loss graph
        :return:
        """
        self.logger.debug('--> Define triplet loss')
        # positive distances
        self.pos_distances = self._pos_distances()

        # negative distances
        self.neg_distances = self._neg_distances()

        loss = tf.maximum(self.pos_distances - self.neg_distances + self.margin,
                          0.0, name='pair_loss')
        self.loss = tf.reduce_sum(loss, name='triplet_loss')

    def _create_train_ops(self):
        """
        Train operations
        :return:
        """
        self.logger.debug('--> Define training operators')
        optimizer = self._build_optimizer(self.learning_rate)
        ops = [optimizer.minimize(self.loss)]
        with tf.control_dependencies(ops):
            self.train_ops = ops + self._clip_by_norm_op()

    def _clip_by_norm_op(self):
        """
        Clip operation by norm
        :return:
        """
        self.logger.debug('----> Define clip norm operators (regularization)')
        return [
            tf.compat.v1.assign(self.user_embeddings,
                                tf.clip_by_norm(self.user_embeddings,
                                                self.clip_norm,
                                                axes=[1])),
            tf.compat.v1.assign(self.item_embeddings,
                                tf.clip_by_norm(self.item_embeddings,
                                                self.clip_norm,
                                                axes=[1]))]

    def _build_optimizer(self, lr):
        """ Builds an optimizer instance from internal parameter values.
        Default to AdamOptimizer if not specified.

        :returns: Optimizer instance from internal configuration.
        """
        self.logger.debug('----> Define optimizer')
        if self.optimizer == self.ADADELTA:
            return tf.compat.v1.train.AdadeltaOptimizer()
        if self.optimizer == self.SGD:
            return tf.compat.v1.train.GradientDescentOptimizer(lr)
        elif self.optimizer == self.ADAM:
            return tf.compat.v1.train.AdamOptimizer(lr)
        else:
            raise HuitreError(f'Unknown optimizer type {self.optimizer}')

    def _pos_distances(self):
        raise NotImplementedError('_pos_distances method should be '
                                  'implemented in concrete model')

    def _neg_distances(self):
        raise NotImplementedError('_neg_distances method should be '
                                  'implemented in concrete model')

    def _create_score_items(self):
        raise NotImplementedError('_create_score_items method should be '
                                  'implemented in concrete model')

    def _build_eval_feedict(self, user_ids):
        return {self.score_user_ids: user_ids}

    def _topk_items(self, feed_dict, k=50):
        """
        Get top k items for the list of users.
        :param feed_dict:
        :param k:
        :return:
        """
        # get back user_ids to be scored
        user_ids = feed_dict[self.score_user_ids]
        _, topk = self.sess.run(
            tf.nn.top_k(self.scores, k),
            feed_dict=feed_dict)
        return dict(zip(user_ids, topk))

    def _get_relations(self, inputs, relations_tf):
        test_relations_path = os.path.join(inputs['extracted_relations_path'],
                                           'test_relations.npz')
        if not os.path.exists(test_relations_path):
            self.logger.info(f'Extracting test relations to {test_relations_path}')
            relations = []
            user_ids = []
            item_ids = []
            for interactions in toolz.partition_all(inputs['chunk_size'],
                                                    inputs['test_interactions']):
                # build input feed_dict
                chunk_user_ids = [uid for uid, _ in interactions]
                chunk_item_ids = [iid for _, iid in interactions]
                feed_dict = self._analyse_feed_dict(chunk_user_ids,
                                                    chunk_item_ids)
                chunk_relations = self.sess.run(relations_tf,
                                                feed_dict=feed_dict)
                user_ids = [*user_ids, *chunk_user_ids]
                item_ids = [*item_ids, *chunk_item_ids]
                relations = [*relations, *chunk_relations]
                n_relations = len(relations)
                if n_relations % 10000 == 0:
                    self.logger.debug(f'Finish extract {n_relations} relations')
            self.logger.debug(f'Finish extract {len(relations)} relations')
            np.savez(test_relations_path, user_ids=user_ids,
                     item_ids=item_ids, relations=relations)
        else:
            self.logger.info(f'Load test relations from {test_relations_path}')
            data = np.load(test_relations_path, allow_pickle=True)
            user_ids = data['user_ids']
            item_ids = data['item_ids']
            relations = data['relations']
        user_ids = np.array(user_ids)
        item_ids = np.array(item_ids)
        relations = np.array(relations)
        return user_ids, item_ids, relations

    def get_relations_tf(self):
        pass

    def _analyse_feed_dict(self, user_ids, item_ids):
        pass

    def _meta_match(self, user_ids, item_ids, user_meta, item_meta, best_relation_indexes):
        num_match = 0
        for rid1, rid2, _ in best_relation_indexes:
            uid1 = user_ids[rid1]
            iid1 = item_ids[rid1]
            uid2 = user_ids[rid2]
            iid2 = item_ids[rid2]
            meta1 = set(user_meta[uid1]).intersection(set(item_meta[iid1]))
            meta2 = set(user_meta[uid2]).intersection(set(item_meta[iid2]))
            if len(meta1.intersection(meta2)) > 0:
                num_match += 1
        self.logger.info(f'Matching probability: '
                         f'{num_match * 1.0 / len(best_relation_indexes)}')
