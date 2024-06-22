import os
import importlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from files import create_results_folder


def load_model(model_config):
    model_module = importlib.import_module(f"src.models.{model_config.name}")
    model = getattr(model_module, model_config.name)(model_config)
    model = model.to(model_config.device)
    return model


def run_experiment(model_config, data_config, features_config):
    create_results_folder(model_config, data_config, features_config)

    if model_config.type == "RF":
        # rf_classifier = RandomForestClassifier(random_state=32)
        # rf_classifier.fit(X_train, y_train)
        # y_pred_test = rf_classifier.predict(X_test)
        # accuracy_test = accuracy_score(y_test, y_pred_test)
        # print(f"El accuracy de test es de {accuracy_test}")
        pass

    elif model_config.type == "NN":
        pass

    