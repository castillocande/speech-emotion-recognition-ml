import os
import numpy as np
import tensorflow as tf
from src.pipeline import run_experiment
from src.files import read_configs


seed_value = 32
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def main(): 
    model_config, data_config, features_config = read_configs()
    run_experiment(model_config, data_config, features_config) 
    

if __name__ == "__main__":
    main()