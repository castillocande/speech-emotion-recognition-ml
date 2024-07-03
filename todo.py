import argparse
import os
import datetime
import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Input, Dropout, LSTM, TimeDistributed, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
import opensmile
from glob import glob
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from importlib.machinery import SourceFileLoader
from types import ModuleType
import importlib

emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


class DataLoader:
    """Clase para cargar y procesar datos de audio."""

    def __init__(self):
        print("DataLoader inicializado")

    def segment_audio2(self, audio, num_frames, top_db=30):
        """Segmenta el audio en varios segmentos después de recortar los silencios."""
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        segment_length = len(trimmed_audio) // num_frames
        segments = []

        for i in range(num_frames):
            start = i * segment_length
            end = (i + 1) * segment_length
            segment = trimmed_audio[start:end]
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)), "constant")
            segments.append(segment)

        return segments

    def process_dataset(self, data_path, save_path, n_segments=1):
        """Procesa el dataset y guarda las características extraídas."""
        files = glob(data_path)
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        features_list = []
        labels_list = []

        for file in files:
            audio, sr = sf.read(file)
            segments = self.segment_audio2(audio, n_segments)
            for segment in segments:
                segment_file = "temp_segment.wav"
                sf.write(segment_file, segment, sr)
                features = smile.process_file(segment_file)
                features_list.append(features.values.flatten())
                labels_list.append(int(os.path.basename(file).split('-')[2]))
        os.remove(segment_file)

        x_ = np.array(features_list)
        y_ = np.array(labels_list)
        y_reshaped = y_[:, np.newaxis]
        dataset = np.concatenate((x_, y_reshaped), axis=1)

        actors = np.array([int(os.path.dirname(path)[-2:]) for path in files])
        np.save(save_path, dataset)
        actors_save_path = self.define_actors_path(save_path)
        np.save(actors_save_path, actors)
        print("Guardado el dataset en", save_path)
        print("Guardado los actores en", actors_save_path)
        return np.load(save_path)

    def get_dataset(self, dataset_path_list, n_segments):
        """Carga el dataset y lo prepara para su uso."""
        dataset = np.load(dataset_path_list[0])
        actors_path = self.define_actors_path(dataset_path_list[0])
        actors = np.load(actors_path)

        for i in range(1, len(dataset_path_list)):
            dataset = np.concatenate((dataset, np.load(dataset_path_list[i])))
            actors_path = self.define_actors_path(dataset_path_list[i])
            actors = np.concatenate((actors, np.load(actors_path)))

        x = dataset[:, :-1]
        y = dataset[:, -1]

        num_samples = x.shape[0] // n_segments
        num_features = x.shape[1]
        x = x.reshape(num_samples, n_segments, num_features)
        y = y.reshape(num_samples, n_segments).mean(axis=1).astype(int)

        print(f"X shape {x.shape}")
        print(f"y shape {y.shape}")
        print(f"actors shape {actors.shape}")
        return x, y, actors

    def split_dataset(self, x, y, test_size=0.2, actors=[]):
        """Divide el dataset en conjuntos de entrenamiento y prueba."""
        if len(actors) > 0:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_idx, test_idx = next(gss.split(x, y, actors))
            X_train, X_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            actors_train = actors[train_idx]
            actors_test = actors[test_idx]

            train_shuffle_idx = np.random.permutation(len(X_train))
            test_shuffle_idx = np.random.permutation(len(X_test))

            X_train, y_train, actors_train = X_train[train_shuffle_idx], y_train[train_shuffle_idx], actors_train[train_shuffle_idx]
            X_test, y_test, actors_test = X_test[test_shuffle_idx], y_test[test_shuffle_idx], actors_test[test_shuffle_idx]

        else:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
            actors_train = None
            actors_test = None

        return X_train, X_test, y_train, y_test, actors_train, actors_test

    def create_batches(self, data, labels, batch_size):
        """Crea lotes de datos y etiquetas para el entrenamiento."""
        num_samples = data.shape[0]
        indices = np.random.permutation(num_samples)
        batch_data = []
        batch_labels = []

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data.append(data[batch_indices])
            batch_labels.append(labels[batch_indices])

        return batch_data, batch_labels

    @staticmethod
    def define_actors_path(dataset_path):
        """Define la ruta de los actores a partir de la ruta del dataset."""
        file_name = os.path.basename(dataset_path).split('_')
        file_name[-1] = file_name[-1].split('.')[0]
        actors_save_path = f"data/{file_name[0]}_{file_name[1]}_actors_{file_name[3]}.npy"
        return actors_save_path


class CNNModel:
    """Clase para definir, entrenar y evaluar el modelo CNN."""

    def __init__(self, img_width=224, img_height=224, batch_size=32, epochs=50):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self.create_model((img_width, img_height, 3), 8)

    def create_model(self, input_shape, num_classes):
        """Crea el modelo CNN."""
        inputs = Input(shape=input_shape)
        x = Conv2D(8, (3, 3), activation="relu")(inputs)
        x = AveragePooling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation="relu")(x)
        x = AveragePooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation="relu")(x)
        x = AveragePooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(2048, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(2048, activation="relu")(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs, outputs)
        return model

    def compile_model(self, learning_rate=0.0001, momentum=0.8):
        """Compila el modelo."""
        optimizer = RMSprop(learning_rate=learning_rate, momentum=momentum)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, train_dir, val_dir):
        """Entrena el modelo usando los datos de entrenamiento y validación."""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.125,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.125)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training"
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation"
        )

        self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            validation_data=val_generator,
            validation_steps=val_generator.samples // self.batch_size,
            epochs=self.epochs
        )

    def evaluate(self, test_dir):
        """Evalúa el modelo usando los datos de prueba."""
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode="categorical"
        )

        test_loss, test_accuracy = self.model.evaluate(test_generator, steps=test_generator.samples // self.batch_size)
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")


class LSTM_Autoencoder_RNN:
    """Clase para definir, entrenar y evaluar el modelo LSTM Autoencoder RNN."""

    def __init__(self, input_shape, lstm_units=128, dropout_rate=0.3, latent_dim=64, optimizer_autoencoder=Adam(), optimizer_rnn=SGD(learning_rate=0.01, momentum=0.9), patience=4, num_classes=8):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        self.optimizer_autoencoder = optimizer_autoencoder
        self.optimizer_rnn = optimizer_rnn
        self.patience = patience
        self.num_classes = num_classes

        self.autoencoder, self.encoder = self.build_autoencoder()
        self.rnn_model = self.build_rnn_model()

    def build_autoencoder(self):
        """Construye el modelo Autoencoder."""
        inputs = Input(shape=self.input_shape)
        flat_inputs = TimeDistributed(Flatten())(inputs)
        encoded = TimeDistributed(Dense(self.latent_dim, activation="relu"))(flat_inputs)
        decoded = TimeDistributed(Dense(np.prod(self.input_shape[1:]), activation="relu"))(encoded)
        decoded = TimeDistributed(Reshape(self.input_shape[1:]))(decoded)
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer=self.optimizer_autoencoder, loss="mse")
        encoder = Model(inputs, encoded)
        return autoencoder, encoder

    def build_rnn_model(self):
        """Construye el modelo RNN."""
        model = Sequential()
        model.add(LSTM(self.lstm_units, return_sequences=False, input_shape=(self.input_shape[0], self.latent_dim)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.num_classes, activation="softmax"))
        model.compile(optimizer=self.optimizer_rnn, loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def train_autoencoder(self, X_train, X_valid, epochs=40, batch_size=32, verbose=1):
        """Entrena el autoencoder."""
        self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, X_valid), verbose=verbose)

    def transform(self, X):
        """Transforma los datos usando el encoder."""
        return self.encoder.predict(X)

    def train_rnn(self, X_train, y_train, X_valid, y_valid, epochs=300, batch_size=32, verbose=1):
        """Entrena el modelo RNN."""
        early_stopping = EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        self.history = self.rnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid), callbacks=[early_stopping], verbose=verbose)

    def evaluate_rnn(self, X_test, y_test):
        """Evalúa el modelo RNN."""
        return self.rnn_model.evaluate(X_test, y_test)

    def plot_learning_curves(self):
        """Grafica las curvas de aprendizaje."""
        history_dict = self.history.history
        loss = history_dict["loss"]
        val_loss = history_dict["val_loss"]
        accuracy = history_dict["accuracy"]
        val_accuracy = history_dict["val_accuracy"]

        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, "bo", label="Training accuracy")
        plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.show()


class rnnLSTM:
    """Clase para definir, entrenar y evaluar el modelo RNN LSTM."""

    def __init__(self, X_train, y_train, lr=0.001, patience=3, dropout_rate=0.5, momentum=0.9):
        self.early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(y_train.shape[1], activation="softmax"))
        self.model.compile(optimizer=SGD(learning_rate=lr, momentum=momentum), loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, X_train, y_train, X_valid, y_valid, epochs=10, batch_size=32, random_seed=42):
        """Entrena el modelo RNN LSTM."""
        tf.random.set_seed(random_seed)
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid), callbacks=[self.early_stopping])

    def evaluate(self, X_test, y_test):
        """Evalúa el modelo RNN LSTM."""
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        return test_loss, test_accuracy

    def plot_learning_curves(self):
        """Grafica las curvas de aprendizaje."""
        history_dict = self.history.history
        loss = history_dict["loss"]
        val_loss = history_dict["val_loss"]
        accuracy = history_dict["accuracy"]
        val_accuracy = history_dict["val_accuracy"]

        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, "bo", label="Training accuracy")
        plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.show()


def extract_spectrogram(audio_path, sr=22050, n_mels=224, fmax=8000):
    """Extrae un espectrograma de un archivo de audio."""
    y, sr = librosa.load(audio_path, sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram


def save_spectrogram(spectrogram, save_path):
    """Guarda un espectrograma en un archivo."""
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(spectrogram, sr=22050, x_axis="time", y_axis="mel", fmax=8000)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def process_files(files, output_path):
    """Procesa archivos de audio y guarda sus espectrogramas en la ruta de salida especificada."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for audio_path in files:
        filename = os.path.basename(audio_path)
        emotion_code = filename.split('-')[2]
        emotion = emotion_labels.get(emotion_code, "unknown")
        class_dir = os.path.join(output_path, emotion)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        spectrogram = extract_spectrogram(audio_path)
        save_path = os.path.join(class_dir, filename.replace(".wav", ".png"))
        save_spectrogram(spectrogram, save_path)


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


def run_experiment(model_config, data_config, features_config):
    """Ejecuta el experimento completo."""
    create_results_folder(model_config, data_config, features_config)

    data_loader = DataLoader()
    X_train, X_test, y_train, y_test, actors_train, actors_test = data_loader.split_dataset(
        *data_loader.get_dataset(data_config.dataset_path_list, data_config.n_segments),
        test_size=data_config.test_size
    )

    if model_config.type == "NN":
        input_shape = X_train.shape[1:]
        model = load_model(model_config, input_shape)

        if isinstance(model, CNNModel):
            model.compile_model()
            model.train(data_config.train_dir, data_config.val_dir)
            model.evaluate(data_config.test_dir)
        else:
            pass


def main():
    """Función principal para ejecutar el experimento."""
    model_config, data_config, features_config = read_configs()
    run_experiment(model_config, data_config, features_config)


if __name__ == "__main__":
    main()
