import os.path
import numpy as np

from huitre.logging import get_logger


class Analyzer:
    """
    Analyzer is responsible to analyse the quality of
    a given model
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
        self.extracted_relations_path = params['extracted_relations_path']
        self.cosine_similarities_path = os.path.join(
            params['cache']['path'], 'fo',
            f'test_relations_best_similarities-{params["training"]["model"]["type"]}.pkl')
        self.pref_items = params['training']['ref_user_items']['train+valid']
        self.max_num_prefs = params['training']['model']['params'].get(
            'max_num_prefs', 50)
        self.test_interactions = params['training']['ref_user_items']['test_interactions']
        self.user_meta = params['training']['ref_user_items']['user_meta']
        self.item_meta = params['training']['ref_user_items']['item_meta']
        self.random_state = params['dataset']['random_state']
        self.n_interactions_in_chunk = params['analyse'].get(
            'n_interactions_in_chunk', 100)
        self.num_interactions = params['analyse'].get(
            'num_interactions', -1)
        self.random_match = params['analyse'].get('random_match', False)

    def do_analyse(self):
        inputs = {
            'extracted_relations_path': self.extracted_relations_path,
            'chunk_size': self.n_interactions_in_chunk,
            'test_interactions': self.test_interactions,
            'user_meta': self.user_meta,
            'item_meta': self.item_meta,
            'cosine_similarities_path': self.cosine_similarities_path,
            'num_interactions': self.num_interactions,
            'random_state': self.random_state
        }
        if self.random_match is False:
            self.model.analyse(inputs)
        else:
            self.do_random_match(inputs)

    @classmethod
    def do_random_match(cls, inputs):
        logger = get_logger()
        test_interactions = inputs['test_interactions']
        rng = np.random.RandomState(inputs['random_state'])
        rng.shuffle(test_interactions)
        test_interactions = test_interactions[:inputs['num_interactions']]
        user_meta = inputs['user_meta']
        item_meta = inputs['item_meta']
        num_match = 0
        for uid1, iid1 in test_interactions:
            rand_idx = np.random.choice(inputs['num_interactions'])
            uid2, iid2 = test_interactions[rand_idx]
            meta1 = set(user_meta[uid1]).intersection(set(item_meta[iid1]))
            meta2 = set(user_meta[uid2]).intersection(set(item_meta[iid2]))
            if len(meta1.intersection(meta2)) > 0:
                num_match += 1
        logger.info(f'Random matching probability: '
                    f'{num_match * 1.0 / len(test_interactions)}')
