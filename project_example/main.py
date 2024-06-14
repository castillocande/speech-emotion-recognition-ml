import logging
import logging.config
from os.path import join
from datetime import datetime

from src.pipeline import run_experiment
from src.utils.files import read_configs, set_seed
from src.utils.logger import set_logger_level

#seed = 1234 # set seed for reproducibility
set_seed(seed)

if __name__ == '__main__':
    model, data, features = read_configs()
    logfilename = '_'.join([cfg.config_name for cfg in [model, data, features]])
    logging.config.fileConfig(join('configs', 'logging.conf'), defaults={'logfilename': join('logs',logfilename)})    
    logger = logging.getLogger('src')
    set_logger_level(logger,'DEBUG')
    logger.info(f'Start experiment. Logfile name: {logfilename}')
    run_experiment(model, data, features)