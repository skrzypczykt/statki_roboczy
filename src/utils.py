import datetime
import os

import yaml


def initialize_output_dir(params):
    tz = datetime.timezone(datetime.timedelta(hours=2))
    ft = "%Y-%m-%d_%H_%M"
    now = datetime.datetime.now(tz=tz).strftime(ft)
    results_dirpath = os.path.join(params['results_dir'], now)
    os.mkdir(results_dirpath)
    with open(os.path.join(results_dirpath, "config.yaml"), "w") as fp:
        yaml.dump(params, fp)
    return results_dirpath