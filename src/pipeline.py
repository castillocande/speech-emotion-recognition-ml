import importlib
from src.files import create_results_folder


def load_model(model_config):
    model_module = importlib.import_module(f"src.models.{model_config.name}")
    model = getattr(model_module, model_config.name)(model_config)
    return model


def run_experiment(model_config, data_config, features_config):
    create_results_folder(model_config, data_config, features_config)

    if model_config.type == "RF":
        pass

    elif model_config.type == "NN":
        model = load_model(model_config)
        if model_config.name == "RNN":
            pass
        elif model_config.name == "CNN":
            model.create_model(model_config.input_shape, 8)
            model.compile_model()
            model.train_model()
            model.evaluate_model()