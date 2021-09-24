from argparse import ArgumentParser

# -p opt specification (train, evaluate and denoise).
OPT_PARAMS = {
    'dest': 'configuration',
    'default': 'huitre:unet',
    'type': str,
    'action': 'store',
    'help': 'JSON filename that contains params'
}

# -a opt specification (train, evaluate and denoise).
OPT_VERBOSE = {
    'action': 'store_true',
    'help': 'Shows verbose logs'
}


def create_argument_parser():
    """ Creates overall command line parser for huitre.

    :returns: Created argument parser.
    """
    parser = ArgumentParser(prog='huitre')
    subparsers = parser.add_subparsers()
    subparsers.dest = 'command'
    subparsers.required = True

    _create_train_parser(subparsers.add_parser)
    _create_eval_parser(subparsers.add_parser)
    _create_analyse_parser(subparsers.add_parser)
    return parser


def _create_train_parser(parser_factory):
    """ Creates an argparser for training command

    :param parser_factory: Factory to use to create parser instance.
    :returns: Created and configured parser.
    """
    parser = parser_factory('train',
                            help='Train a recommendation model')
    _add_common_options(parser)
    return parser


def _create_eval_parser(parser_factory):
    """ Creates an argparser for evaluation command

    :param parser_factory: Factory to use to create parser instance.
    :returns: Created and configured parser.
    """
    parser = parser_factory(
        'eval',
        help='Evaluate a model on the musDB test dataset')
    _add_common_options(parser)
    return parser


def _create_analyse_parser(parser_factory):
    """ Creates an argparser for evaluation command

    :param parser_factory: Factory to use to create parser instance.
    :returns: Created and configured parser.
    """
    parser = parser_factory(
        'analyse',
        help='Analyse a model on a test dataset')
    _add_common_options(parser)
    return parser


def _add_common_options(parser):
    """ Add common option to the given parser.

    :param parser: Parser to add common opt to.
    """
    parser.add_argument('-p', '--params_filename', **OPT_PARAMS)
    parser.add_argument('--verbose', **OPT_VERBOSE)
