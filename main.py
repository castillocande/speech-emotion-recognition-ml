from src.pipeline import run_experiment
from src.files import read_configs


def main(): 
    model_config, data_config, features_config = read_configs()
    run_experiment(model_config, data_config, features_config) 
    

if __name__ == "__main__":
    main()