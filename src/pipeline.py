import importlib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.data_loader import DataLoader
from src.files import create_results_folder
from src import data_loader, aux_functions
from src.models import RandomForest, lstm, autoencoder_lstm, cnn


speech1_dataset_dev = "./data/1frame_Speech_dataset_dev.npy"
speech3_dataset_dev = "./data/3frame_Speech_dataset_dev.npy"

song1_dataset_dev = "./data/1frame_Song_dataset_dev.npy"
song3_dataset_dev = "./data/3frame_Song_dataset_dev.npy"

speech1_dataset_test = "./data/1frame_Speech_dataset_test.npy"
speech3_dataset_test = "./data/3frame_Speech_dataset_test.npy"

song1_dataset_test = "./data/1frame_Song_dataset_test.npy"
song3_dataset_test = "./data/3frame_Song_dataset_test.npy"


def run_experiment(model_config, data_config, features_config):
    create_results_folder(model_config, data_config, features_config)

    if model_config.type == "RF":
        DL = data_loader.DataLoader()
        X_dev, y_dev, actors_dev = DL.get_dataset([song1_dataset_dev, speech1_dataset_dev], 1)
        X_train, X_valid, y_train, y_valid, actors_train, actors_valid = DL.split_dataset(X_dev, y_dev, test_size=0.2, actors=actors_dev)
        X_test, y_test, actors_test = DL.get_dataset([song1_dataset_test, speech1_dataset_test], 1)
        X_train, y_train = aux_functions.SMOTE_(X_train, y_train)

        RF_hyperparams_save_path = "./results/RF_hyperparameters.npy" 
        RF_hyperparams = np.load(RF_hyperparams_save_path, allow_pickle=True)
        RF = RandomForest.RandomForest(n_estimators=RF_hyperparams[0],
                             max_features=RF_hyperparams[1],
                             max_depth=RF_hyperparams[2],
                             min_samples_split=RF_hyperparams[3],
                             min_samples_leaf=RF_hyperparams[4],
                             bootstrap=RF_hyperparams[5],
                             criterion=RF_hyperparams[6])
        
        RF.fit(X_train, y_train, X_valid, y_valid)
        y_predict = RF.predict(X_test)

        acc = accuracy_score(y_test, y_predict)
        print(f" trees --> test Accuracy {acc}")

        conf_matrix = confusion_matrix(y_test, y_predict)
        print("Confusion Matrix:")
        print(conf_matrix)

        print("\nClassification Report:")
        print(classification_report(y_test, y_predict))


    elif model_config.type == "LSTM":
        X_dev, y_dev, actors_dev = DL.get_dataset([song3_dataset_dev, speech3_dataset_dev], 3)
        X_train, X_valid, y_train, y_valid, actors_train, actors_valid = DL.split_dataset(X_dev, y_dev, test_size=0.2, actors=actors_dev)
        X_test, y_test, actors_test = DL.get_dataset([song3_dataset_test, speech3_dataset_test], 3)

        X_train, y_train = aux_functions.SMOTE_(X_train, y_train)
        X_train = aux_functions.normalization(X_train)
        X_valid = aux_functions.normalization(X_valid)
        X_test = aux_functions.normalization(X_test)

        y_train_ohe  = aux_functions.one_hot_encoder(y_train)
        y_valid_ohe = aux_functions.one_hot_encoder(y_valid)
        y_test_ohe = aux_functions.one_hot_encoder(y_test)
        y_dev_ohe =aux_functions. one_hot_encoder(y_dev)

        RNN_hyperparams_save_path = "./results/RNN_hyperparameters.npy" 
        hyperparams = np.load(RNN_hyperparams_save_path)

        RNN = lstm.rnnLSTM(X_train, y_train_ohe, lr=hyperparams[1], dropout_rate=hyperparams[2], patience=3, momentum=hyperparams[3])
        RNN.train(X_train, y_train_ohe, X_valid, y_valid_ohe, epochs=500, batch_size=int(hyperparams[0]))
        RNN.plot_learning_curves()

        print("test")
        test_loss, test_accuracy = RNN.evaluate(X_test, y_test_ohe)
        print(f'Test Loss: {test_loss}')
        print(f'Test Accuracy: {test_accuracy}')


    elif model_config.type == "AE_LSTM":
        DL = data_loader.DataLoader()
        X_dev, y_dev, actors_dev = DL.get_dataset([song3_dataset_dev, speech3_dataset_dev], 3)
        X_train, X_valid, y_train, y_valid, actors_train, actors_valid = DL.split_dataset(X_dev, y_dev, test_size=0.2, actors=actors_dev)
        X_test, y_test, actors_test = DL.get_dataset([song3_dataset_test, speech3_dataset_test], 3)

        X_train, y_train = aux_functions.SMOTE_(X_train, y_train)
        X_train = aux_functions.normalization(X_train)
        X_valid = aux_functions.normalization(X_valid)
        X_test = aux_functions.normalization(X_test)

        y_train_ohe  = aux_functions.one_hot_encoder(y_train)
        y_valid_ohe = aux_functions.one_hot_encoder(y_valid)
        y_test_ohe = aux_functions.one_hot_encoder(y_test)
        y_dev_ohe =aux_functions. one_hot_encoder(y_dev)

        input_shape = (X_train.shape[1], X_train.shape[2])  
        lstm_autoencoder_rnn = autoencoder_lstm.LSTM_Autoencoder_RNN(input_shape=input_shape)
        lstm_autoencoder_rnn.train_autoencoder(X_train, X_valid)

        X_train_encoded = lstm_autoencoder_rnn.transform(X_train)
        X_valid_encoded = lstm_autoencoder_rnn.transform(X_valid)
        X_test_encoded = lstm_autoencoder_rnn.transform(X_test)

        lstm_autoencoder_rnn.train_rnn(X_train_encoded, y_train_ohe, X_valid_encoded, y_valid_ohe)
        lstm_autoencoder_rnn.plot_learning_curves()

        print("test")
        test_loss, test_accuracy = lstm_autoencoder_rnn.evaluate_rnn(X_test_encoded, y_test_ohe)
        print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')


    elif model_config.type == "CNN":
        train_dir = "data/spectrograms/combined"
        test_dir = "data/spectrograms_test/combined"
        input_shape = (224, 224, 3)
        num_classes = 8
    
        cnn_model = cnn.CNN(train_dir, test_dir)
        cnn_model.create_model(input_shape, num_classes)
        cnn_model.compile_model()
        cnn_model.train_model()
        cnn_model.evaluate_model()
       