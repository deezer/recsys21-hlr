import logging
from os import environ


_FORMAT = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'


class _LoggerHolder(object):
    """
    Logger singleton instance holder.
    """
    INSTANCE = None


def get_tensorflow_logger():
    """
    """
    from tensorflow.compat.v1 import logging
    return logging


def get_logger():
    """
    Returns library scoped logger.
    :returns: Library logger.
    """
    if _LoggerHolder.INSTANCE is None:
        formatter = logging.Formatter(_FORMAT)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger('huitre')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        _LoggerHolder.INSTANCE = logger
    return _LoggerHolder.INSTANCE


def enable_tensorflow_logging():
    """ Enable tensorflow logging. """
    environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf_logger = get_tensorflow_logger()
    tf_logger.set_verbosity(tf_logger.INFO)
    logger = get_logger()
    logger.setLevel(logging.DEBUG)


def enable_logging():
    """ Configure default logging. """
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf_logger = get_tensorflow_logger()
    tf_logger.set_verbosity(tf_logger.ERROR)
