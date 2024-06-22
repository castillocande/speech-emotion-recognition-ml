import os
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def create_results_folder(*config_settings): # recibe model_config, data_config y features_config
    names = [config_settings[i].name for i in range(len(config_settings))]
    base = os.path.join("results", *names)
    now = datetime.datetime.now()
    os.makedirs(base, exist_ok=True)


def run_experiment(model_config, data_config, features_config):
    create_results_folder(model_config, data_config, features_config)

    if model_config.type == "RF":
        rf_classifier = RandomForestClassifier(random_state=32)
        rf_classifier.fit(X_train, y_train)
        y_pred_test = rf_classifier.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        print(f"El accuracy de test es de {accuracy_test}")

    elif model_config.type == "NN":
        pass

    