import os
import numpy as np
import pickle
from huitre import HuitreError
from huitre.evaluators import Evaluator
from huitre.logging import get_logger


class Recommender:
    """
    Recommender is responsible to generate recommendation from
    a given model or to evaluate this model
    """
    def __init__(self, sess, model, params):
        """
        Initialization a recommender
        :param sess: global session
        :param model: model to be trained
        :param params: hyperparameters for training
        """
        self.sess = sess
        self.model = model
        self.params = params
        self.model_dir = params['training']['model_dir']
        self.model_type = params['training']['model']['type']
        self.random_state = params['dataset']['random_state']
        self.n_eval_users = params['eval'].get('n_users', 30000)
        self.n_cohort = params['eval'].get('n_cohort', 5)
        self.eval_acc_metrics = params['eval'].get('acc', True)
        self.logger = get_logger()

    def eval(self, corpus='test'):
        """
        Evaluate the model
        :return
        """
        test_user_items = self.params['training']['ref_user_items'][corpus]
        n_users_in_chunk = self.params['eval'].get(
            'n_users_in_chunk', 50)
        # evaluator
        evaluator = Evaluator(
            self.params['eval'],
            ref_user_items=test_user_items)

        reco_dir = self.params['eval']['reco_dir']
        items_metadata = self.params['eval']['items_metadata']

        user_ids = test_user_items.keys()
        # remove users that do not have any interactions in test
        user_ids = [uid for uid in user_ids if len(test_user_items[uid]) > 0]
        rng = np.random.RandomState(self.random_state)
        final_scores = []
        rng.shuffle(user_ids)
        for eval_idx in range(self.n_cohort):
            if eval_idx >= 0:
                cohort_cache = os.path.join(reco_dir,
                                            f'cohort_{eval_idx+1}.pkl')
                if not os.path.exists(cohort_cache):
                    eval_user_ids = user_ids[eval_idx*self.n_eval_users:
                                             (eval_idx+1)*self.n_eval_users]
                    self.logger.debug(f'*********** Eval num: {eval_idx + 1} for '
                                      f'{len(eval_user_ids)} users ***********')

                    if self.model_type == 'transcf':
                        user_embeddings, item_embeddings = self.model.embeddings()
                        self.model.build_ui_relations(eval_user_ids,
                                                      user_embeddings,
                                                      item_embeddings)
                    # get recommended items
                    reco_items = self.recommend(user_ids=eval_user_ids,
                                                num_items=evaluator.max_k,
                                                corpus=corpus,
                                                n_users_in_chunk=n_users_in_chunk)
                    pickle.dump(reco_items, open(cohort_cache, 'wb'))
                else:
                    reco_items = pickle.load(open(cohort_cache, 'rb'))
                # get scores
                if self.eval_acc_metrics is True:
                    scores = evaluator.eval(reco_items)
                else:
                    scores = evaluator.eval_non_acc(reco_items,
                                                    items_metadata)
                final_scores.append(scores)
        return final_scores

    def recommend(self, user_ids, num_items,
                  corpus='test', n_users_in_chunk=50):
        """
        Recommend num_items for a given user list
        :param user_ids:
        :param num_items:
        :param corpus:
        :param n_users_in_chunk:
        :return
        """
        if corpus == 'valid':
            train_user_items = self.params['training']['ref_user_items']['train']
        elif corpus == 'test':
            train_user_items = self.params['training']['ref_user_items']['train+valid']
        else:
            raise HuitreError(f'Eval corpus should be `valid` or `test`. '
                              f'Recieve {corpus}')
        # get recommend items
        reco_items = self.model.recommend(
            users=user_ids,
            excluded_items=train_user_items,
            num_items=num_items,
            n_users_in_chunk=n_users_in_chunk
        )
        return reco_items
