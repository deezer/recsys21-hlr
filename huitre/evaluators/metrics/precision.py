import numpy as np
from huitre.evaluators.metrics.metric import Metric


class PRECISION(Metric):
    """
    precision@k score metric.
    """
    def __str__(self):
        return f'precision@{self.k}'

    @classmethod
    def precision_at_k(cls, r, k):
        """
        Precision at k
        :param r:
        :param k:
        :return:
        """
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        # return np.mean(r)
        return sum(r) / len(r)

    def eval(self, reco_items, ref_user_items):
        """
        Compute the Top-K PRECISION
        :param reco_items: reco items dictionary
        :param ref_user_items:
        :return: precision@k
        """
        prec_metric = []
        for user_id, top_items in reco_items.items():
            ref_set = ref_user_items.get(user_id, set())
            user_hits = np.array([1 if it in ref_set else 0 for it in top_items],
                                 dtype=np.float32)
            prec_metric.append(self.precision_at_k(user_hits, self.k))
        return np.mean(prec_metric)
