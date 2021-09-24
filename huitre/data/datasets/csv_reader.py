import os
import pandas as pd
from huitre.logging import get_logger


def read_data(params):
    """
    Fetch interactions from file
    :param params: dataset parameters
    """
    interaction_path = os.path.join(params['path'],
                                    params['interactions'])
    get_logger().debug(f'Read {params["name"]} data from {interaction_path}')
    # read interaction csv file
    sep = params.get('sep', ',')
    header = params.get('header', None)
    encoding = params.get('encoding', 'utf8')
    data = pd.read_csv(interaction_path,
                       sep=sep,
                       header=header,
                       names=params['col_names'],
                       encoding=encoding)
    data['rating'] = data['rating'].astype(int)
    return data
