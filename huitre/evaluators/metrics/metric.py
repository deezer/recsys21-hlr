class Metric:
    """
    Abstract class for a recommendation metric
    """
    def __init__(self, k, rank_method='local_rank'):
        self.k = k
        self.rank_method = rank_method

    def eval(self, reco_items, ref_user_items):
        """
        Abstract
        :param reco_items:
        :param ref_user_items:
        :return:
        """
        raise NotImplementedError(
            'eval method should be implemented in concrete model')
