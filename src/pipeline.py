import os
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


# def read_results_file():
    # config_filename = 'random_forest_config.pkl'
    # base_directory_path = r"results\random_forest\RAVDESS\egemaps"
    # config_path = os.path.join(base_directory_path, config_filename)

    # with open(config_path, 'rb') as f:
    #     loaded_config = pickle.load(f)

    # print("Loaded configuration:")
    # for key, value in loaded_config.items():
    #     print(f"{key}: {value}")


def create_results_folder(*config_settings): # recibe model_config, data_config y features_config
    names = [config_settings[i].name for i in range(len(config_settings))]
    base = os.path.join("results", *names)
    now = datetime.datetime.now()
    os.makedirs(base, exist_ok=True)
    for param in config_settings:
        configs = {}
        for setting in dir(param):
            configs[setting] = getattr(param, setting)
        with open(os.path.join(base, f'{param.name}_config.pkl'), 'wb') as f:
            pickle.dump(configs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(base, 'run_date.txt'), 'w') as f:
        f.write(str(now))
        f.close
    # read_results_file()
    return base



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

    