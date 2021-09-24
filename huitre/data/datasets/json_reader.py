import os
import json
import pandas as pd
from huitre.logging import get_logger


def read_data(params):
    """
    Fetch interactions from json file
    :param params: dataset parameters
    """
    data_path = params['path']
    interaction_path = os.path.join(data_path,
                                    params['interactions'])
    get_logger().debug(f'Read {params["name"]} data from {interaction_path}')
    # read interaction data from json file and convert to dataframe
    data = []
    with open(interaction_path) as f:
        # for each line in the json file
        for line in f:
            # store the line in the array for manipulation
            record = json.loads(line)
            data.append((record['user_id'], record['business_id'], record['stars']))
    data = pd.DataFrame(data, columns=['org_user', 'org_item', 'rating'])
    return data
