import opensmile
from glob import glob
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit, train_test_split


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