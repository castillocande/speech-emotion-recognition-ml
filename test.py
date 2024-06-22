from src.pipeline import run_experiment
from src.files import read_configs
import argparse
import os
from importlib.machinery import SourceFileLoader
from types import ModuleType


def file_path(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} does not exist or is not a file")
    return path


def read_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=file_path, required=True, help="Path to the model file")
    parser.add_argument("--data", type=file_path, required=True, help="Path to the data file")
    parser.add_argument("--features", type=file_path, required=True, help="Path to the features file")
    args = parser.parse_args()
    model_config = get_config(args.model)
    data_config = get_config(args.data)
    features_config = get_config(args.features)
    return model_config, data_config, features_config


def get_config(path):
    loader = SourceFileLoader("config", path)
    mod = ModuleType(loader.name)
    loader.exec_module(mod)
    for var in ["__name__", "__doc__", "__package__", "__loader__", "__spec__", "__builtins__"]:
        delattr(mod, var)
    return mod


def run_experiment(model, data, features):
    if model.type == "RF":
        pass
    elif model.type == "NN":
        pass


def main(): 
    model_config, data_config, features_config = read_configs()
    run_experiment(model_config, data_config, features_config) 


if __name__ == "__main__":
    main()