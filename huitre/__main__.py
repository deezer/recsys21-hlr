import sys
import warnings

from huitre import HuitreError
from huitre.commands import create_argument_parser
from huitre.configuration import load_configuration
from huitre.logging import (
    enable_logging,
    enable_tensorflow_logging,
    get_logger)


def main(argv):
    try:
        parser = create_argument_parser()
        arguments = parser.parse_args(argv[1:])
        enable_logging()
        if arguments.verbose:
            enable_tensorflow_logging()
        if arguments.command == 'train':
            from huitre.commands.train import entrypoint
        elif arguments.command == 'eval':
            from huitre.commands.eval import entrypoint
        elif arguments.command == 'analyse':
            from huitre.commands.analyse import entrypoint
        else:
            raise HuitreError(
                f'Huitre does not support command {arguments.command}')
        params = load_configuration(arguments.configuration)
        entrypoint(params)
    except HuitreError as e:
        get_logger().error(e)


def entrypoint():
    """ Command line entrypoint. """
    warnings.filterwarnings('ignore')
    main(sys.argv)


if __name__ == '__main__':
    entrypoint()
