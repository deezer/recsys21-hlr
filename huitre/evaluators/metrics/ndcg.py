import numpy as np
from huitre.evaluators.metrics.metric import Metric


class NDCG(Metric):
    """
    nDCG@k score metric.
    """
    @classmethod
    def dcg_at_k(cls, r, k):
        """
        Discounted Cumulative Gain calculation method
        :param r:
        :param k:
        :return: float, DCG value
        """
        assert k >= 1
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.

    def eval(self, reco_items, ref_user_items):
        local_res = []
        global_res = []
        for user_id, top_items in reco_items.items():
            ref_set = ref_user_items.get(user_id, set())
            user_hits = np.array([1 if it in ref_set else 0 for it in top_items],
                                 dtype=np.float32)
            local_ideal_rels = np.array(sorted(user_hits, reverse=True))
            global_ideal_rels = self._global_ideal_rels(ref_set)

            dcg_k = self.dcg_at_k(user_hits, self.k)
            local_ideal_dcg = self.dcg_at_k(local_ideal_rels, self.k)
            if local_ideal_dcg > 0.:
                loc_ndcg = dcg_k / local_ideal_dcg
                local_res.append(loc_ndcg)

            global_ideal_dcg = self.dcg_at_k(global_ideal_rels, self.k)
            if global_ideal_dcg > 0.:
                glob_ndcg = dcg_k / global_ideal_dcg
                global_res.append(glob_ndcg)

        loc_ndcg = np.mean(local_res) if len(local_res) > 0 else 0.
        glob_ndcg = np.mean(global_res) if len(global_res) > 0 else 0.
        return loc_ndcg, glob_ndcg

    def _global_ideal_rels(self, ref_set):
        if len(ref_set) >= self.k:
            ideal_rels = np.ones(self.k)
        else:
            ideal_rels = np.pad(np.ones(len(ref_set)),
                                (0, self.k - len(ref_set)),
                                'constant')
        return ideal_rels

    def __str__(self):
        return f'ndcg@{self.k}'
