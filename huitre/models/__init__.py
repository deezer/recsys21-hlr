from huitre import HuitreError
from huitre.models.cml import CML
from huitre.models.trans.lrml import LRML
from huitre.models.atts.attcml import AttCML
from huitre.models.trans.transcf import TransCF
from huitre.models.hiers.hlr import HLRE
from huitre.models.hiers.jupiter import JUPITER
from huitre.models.bpr import BPR
from huitre.models.recommender import Recommender
from huitre.models.analyzer import Analyzer


SUPPORTED_MODELS = {
    'cml': CML,
    'attcml': AttCML,
    'lrml': LRML,
    'transcf': TransCF,
    'hlre': HLRE,
    'jpte': JUPITER,
    'bpr': BPR
}


class ModelFactory:
    @classmethod
    def generate_model(cls, sess, params, n_users, n_items, command='train'):
        """
        Factory method to generate a model
        :param sess:
        :param params:
        :param n_users:
        :param n_items:
        :param command:
        :return:
        """
        model_type = params['model']['type']
        try:
            # create a new model
            mdl = SUPPORTED_MODELS[model_type](sess=sess,
                                               params=params,
                                               n_users=n_users,
                                               n_items=n_items)
            if command == 'train':
                # build computation graph
                mdl.build_graph()
            elif command == 'eval':
                mdl.restore()
            return mdl
        except KeyError:
            raise HuitreError(f'Currently not support model {model_type}')
