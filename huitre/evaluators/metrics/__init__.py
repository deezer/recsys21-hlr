from huitre import HuitreError
from huitre.evaluators.metrics.ndcg import NDCG
from huitre.evaluators.metrics.hit_rate import HITRATE
from huitre.evaluators.metrics.map import MAP
from huitre.evaluators.metrics.mrr import MRR
from huitre.evaluators.metrics.precision import PRECISION
from huitre.evaluators.metrics.recall import RECALL


_SUPPORTED_METRICS = {
    'ndcg': NDCG,
    'hitrate': HITRATE,
    'map': MAP,
    'mrr': MRR,
    'precision': PRECISION,
    'recall': RECALL
}


def get_metric(name, k, rank_method):
    """
    Get metric object from configuration
    :param name:
    :param k:
    :param rank_method:
    :return:
    """
    if name not in _SUPPORTED_METRICS:
        raise HuitreError(f'Not supported metric `{name}`. '
                          f'Must one of {list(_SUPPORTED_METRICS.keys())}.')
    return _SUPPORTED_METRICS[name](k=k,
                                    rank_method=rank_method)
