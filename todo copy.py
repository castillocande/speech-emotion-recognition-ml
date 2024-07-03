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


class CNN:
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

    def plot_learning_curves(self):
        """Grafica las curvas de aprendizaje del entrenamiento y validación."""
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


def load_model(model_config):
    model_module = importlib.import_module(f"src.models.{model_config.name}")
    model = getattr(model_module, model_config.name)(model_config)
    model = model.to(model_config.device)
    return model


def run_experiment(model_config, data_config, features_config):
    create_results_folder(model_config, data_config, features_config)

    if model_config.type == "RF":
        pass

    elif model_config.type == "NN":
        model = load_model(model_config, model_config.input_dim)
        if model_config.name == "CNN":
            model.compile_model()
            model.train(data_config.train_dir, data_config.val_dir)
            model.evaluate(data_config.test_dir)


def main():
    """Función principal para ejecutar el experimento."""
    model_config, data_config, features_config = read_configs()
    run_experiment(model_config, data_config, features_config)


if __name__ == "__main__":
    main()
