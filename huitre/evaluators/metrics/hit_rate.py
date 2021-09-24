import numpy as np
from huitre.evaluators.metrics.metric import Metric


class HITRATE(Metric):
    """
    recall@k score metric.
    """
    def __str__(self):
        return f'hit_rate@{self.k}'

    def eval(self, reco_items, ref_user_items):
        """
        Compute the Top-K hit_rate
        :param reco_items: reco items dictionary
        :param ref_user_items:
        :return: hit_rate@k
        """

        hit_rate = []
        for user_id, tops in reco_items.items():
            # remove first element of the sequence
            ref_set = ref_user_items.get(user_id, [])
            user_hits = np.array([1 if it in ref_set else 0 for it in tops],
                                 dtype=np.float32)
            hit_rate.append(float(np.sum(user_hits[:self.k])) / len(ref_set))
        return np.mean(hit_rate)
