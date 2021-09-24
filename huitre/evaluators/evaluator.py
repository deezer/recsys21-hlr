import numpy as np
from huitre.evaluators.metrics import get_metric


class Evaluator:
    """
    Evaluator for recommendation algorithms
    """
    def __init__(self, config, ref_user_items):
        """
        Initialize an evaluator
        :param config:
        :param ref_user_items: dictionary of user items
        """
        rank_method = config['rank']
        self.metrics = [get_metric(conf['name'], k, rank_method)
                        for conf in config['metrics']
                        for k in conf['params']['k']]
        self.ref_user_items = ref_user_items
        self.max_k = np.max([k for conf in config['metrics']
                             for k in conf['params']['k']])

    def __str__(self):
        return 'Evaluator: ' + self.metric_str(sep='_')

    def eval(self, reco_items):
        return {
            str(m): m.eval(reco_items, self.ref_user_items)
            for m in self.metrics
        }

    @classmethod
    def eval_non_acc(cls, reco_items, items_metadata):
        items_popularity = items_metadata['popularity']
        recommended_items = set()
        for _, iids in reco_items.items():
            for iid in iids:
                recommended_items.add(iid)
        item_ranks = [items_popularity[iid] for iid in recommended_items]
        return np.median(item_ranks)

    def metric_str(self, sep=','):
        return sep.join([str(m) for m in self.metrics])
