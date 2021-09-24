from huitre import HuitreError
from huitre.data.generators.generator import BatchGenerator

SUPPORTED_GENERATORS = {
    'pairwise': BatchGenerator
}


def generator_factory(interactions, batch_size,
                      num_negatives, random_state,
                      user_items=None,
                      gen_type='pairwise',
                      **kwargs):
    """
    Generator factory
    :param interactions:
    :param batch_size:
    :param num_negatives:
    :param random_state:
    :param user_items: default user items
    :param gen_type:
    :param kwargs:
    :return:
    """
    try:
        return SUPPORTED_GENERATORS[gen_type](
            interactions=interactions,
            batch_size=batch_size,
            num_negatives=num_negatives,
            random_state=random_state,
            user_items=user_items,
            **kwargs)
    except KeyError:
        raise HuitreError(f'Not support generator type: {gen_type}')
