import numpy as np
from huitre.evaluators.metrics.metric import Metric


class MRR(Metric):
    """
    mrr@k score metric.
    """
    def __str__(self):
        return f'mrr@{self.k}'

    @classmethod
    def mrr_at_k(cls, user_hits, k):
        assert k >= 1
        user_hits = np.asarray(user_hits)[:k] != 0
        res = 0.
        for index, item in enumerate(user_hits):
            if item == 1:
                res += 1 / (index + 1)
        return res

    def eval(self, reco_items, ref_user_items):
        """
        Compute the Top-K MRR for a particular user given the predicted scores
        to items.
        :param reco_items: reco items dictionary (contains also metadata
                                                  necessary, e.g. art_ids)
        :param ref_user_items:
        :return: map@k
        """
        mrr_metric = []
        for user_id, top_items in reco_items.items():
            ref_set = ref_user_items.get(user_id, set())
            user_hits = np.array([1 if it in ref_set else 0 for it in top_items],
                                 dtype=np.float32)
            mrr_metric.append(self.mrr_at_k(user_hits, self.k))
        return np.mean(mrr_metric)
