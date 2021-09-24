import numpy as np
from huitre.evaluators.metrics.metric import Metric


class RECALL(Metric):
    """
    recall@k score metric.
    """
    def __str__(self):
        return f'recall@{self.k}'

    @classmethod
    def user_rec(cls, user_hits, ref_len, k):
        score = 0.0
        user_hits = np.asfarray(user_hits)[:k]
        sum_hits = np.sum(user_hits)

        # in the case where the list contains no hit, return score 0.0 directly
        if sum_hits == 0:
            return score
        return float(sum_hits) / ref_len

    def eval(self, reco_items, ref_user_items):
        """
        Compute the Top-K recall
        :param reco_items: reco items dictionary
        :param ref_user_items:
        :return: recall@k
        """
        recall = []
        for user_id, top_items in reco_items.items():
            ref_set = ref_user_items.get(user_id, set())
            user_hits = np.array([1 if it in ref_set else 0 for it in top_items],
                                 dtype=np.float32)
            recall.append(self.user_rec(user_hits, len(ref_set), self.k))
        return np.mean(recall)
