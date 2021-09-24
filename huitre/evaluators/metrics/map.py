import numpy as np
from huitre.evaluators.metrics.metric import Metric


class MAP(Metric):
    """
    map@k score metric.
    """
    def __str__(self):
        return f'map@{self.k}'

    @classmethod
    def precision_at_k(cls, r, k):
        """
        :param r:
        :param k:
        :return:
        """
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    @classmethod
    def ap(cls, r):
        """
        Average precision
        :param r:
        :return:
        """
        r = np.asarray(r) != 0
        out = [cls.precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return np.sum(out) / len(r)

    def eval(self, reco_items, ref_user_items):
        """
        Compute the Top-K MAP for a particular user given the predicted scores
        to items.
        :param reco_items: reco items dictionary (contains also metadata
                                                  necessary, e.g. art_ids)
        :param ref_user_items:
        :return: map@k
        """
        map_metric = []
        for user_id, top_items in reco_items.items():
            ref_set = ref_user_items.get(user_id, set())
            user_hits = np.array([1 if it in ref_set else 0 for it in top_items],
                                 dtype=np.float32)
            map_metric.append(self.ap(user_hits[:self.k]))
        return np.mean(map_metric)
