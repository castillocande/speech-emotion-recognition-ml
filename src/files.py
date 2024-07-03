import argparse
import os
import datetime
import pickle
from importlib.machinery import SourceFileLoader
from types import ModuleType
import importlib


def file_path(path):
    """Valida si una ruta es un archivo."""
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} does not exist or is not a file")
    return path


def read_configs():
    """Lee los archivos de configuración del modelo, los datos y las características."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=file_path, required=True, help="Ruta al archivo del modelo")
    parser.add_argument("--data", type=file_path, required=True, help="Ruta al archivo de datos")
    parser.add_argument("--features", type=file_path, required=True, help="Ruta al archivo de características")
    args = parser.parse_args()
    model_config = get_config(args.model)
    data_config = get_config(args.data)
    features_config = get_config(args.features)
    return model_config, data_config, features_config


def get_config(path):
    """Carga un archivo de configuración."""
    loader = SourceFileLoader("config", path)
    mod = ModuleType(loader.name)
    loader.exec_module(mod)
    for var in ["__name__", "__doc__", "__package__", "__loader__", "__spec__", "__builtins__"]:
        delattr(mod, var)
    return mod


def create_results_folder(*config_settings):
    """Crea una carpeta para guardar los resultados del experimento."""
    names = [config_settings[i].name for i in range(len(config_settings))]
    base = os.path.join("results", *names)
    now = datetime.datetime.now()
    os.makedirs(base, exist_ok=True)
    for param in config_settings:
        configs = {}
        for setting in dir(param):
            configs[setting] = getattr(param, setting)
        with open(os.path.join(base, f"{param.name}_config.pkl"), "wb") as f:
            pickle.dump(configs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(base, "run_date.txt"), "w") as f:
        f.write(str(now))
        f.close()
    return base


def load_model(model_config, input_shape):
    """Carga el modelo basado en la configuración proporcionada."""
    model_module = importlib.import_module(f"src.models.{model_config.name}")
    model_class = getattr(model_module, model_config.name)
    model = model_class(input_shape, **model_config.params)
    return model


# def file_path(path):
#     if not os.path.isfile(path):
#         raise argparse.ArgumentTypeError(f"{path} does not exist or is not a file")
#     return path


# def read_configs():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=file_path, required=True, help="Path to the model file")
#     parser.add_argument("--data", type=file_path, required=True, help="Path to the data file")
#     parser.add_argument("--features", type=file_path, required=True, help="Path to the features file")
#     args = parser.parse_args()
#     model_config = get_config(args.model)
#     data_config = get_config(args.data)
#     features_config = get_config(args.features)
#     return model_config, data_config, features_config


# def get_config(path):
#     loader = SourceFileLoader("config", path)
#     mod = ModuleType(loader.name)
#     loader.exec_module(mod)
#     for var in ["__name__", "__doc__", "__package__", "__loader__", "__spec__", "__builtins__"]:
#         delattr(mod, var)
#     return mod


# # def read_results_file():
#     # config_filename = 'random_forest_config.pkl'
#     # base_directory_path = r"results\random_forest\RAVDESS\egemaps"
#     # config_path = os.path.join(base_directory_path, config_filename)

#     # with open(config_path, 'rb') as f:
#     #     loaded_config = pickle.load(f)

#     # for key, value in loaded_config.items():
#     #     print(f"{key}: {value}")


# def create_results_folder(*config_settings): # recibe model_config, data_config y features_config
#     names = [config_settings[i].name for i in range(len(config_settings))]
#     base = os.path.join("results", *names)
#     now = datetime.datetime.now()
#     os.makedirs(base, exist_ok=True)
#     for param in config_settings:
#         configs = {}
#         for setting in dir(param):
#             configs[setting] = getattr(param, setting)
#         with open(os.path.join(base, f'{param.name}_config.pkl'), 'wb') as f:
#             pickle.dump(configs, f, protocol=pickle.HIGHEST_PROTOCOL)
#     with open(os.path.join(base, 'run_date.txt'), 'w') as f:
#         f.write(str(now))
#         f.close
#     # read_results_file()
#     return base