from tensorflow.keras.layers import LSTM, Dropout, Dense, TimeDistributed, Flatten, Input, Reshape
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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



# class LSTM_Autoencoder_RNN:
#     def __init__(self, input_shape, lstm_units=128, dropout_rate=0.3, latent_dim=64, optimizer_autoencoder=Adam(), optimizer_rnn=SGD(learning_rate=0.01, momentum=0.9), patience=4, num_classes=8):
#         self.input_shape = input_shape
#         self.lstm_units = lstm_units
#         self.dropout_rate = dropout_rate
#         self.latent_dim = latent_dim
#         self.optimizer_autoencoder = optimizer_autoencoder
#         self.optimizer_rnn = optimizer_rnn
#         self.patience = patience
#         self.num_classes = num_classes

#         # Build autoencoder
#         self.autoencoder, self.encoder = self.build_autoencoder()

#         # Build RNN model
#         self.rnn_model = self.build_rnn_model()
    
#     def build_autoencoder(self):
#         inputs = Input(shape=self.input_shape)
        
#         # Encoder
#         flat_inputs = TimeDistributed(Flatten())(inputs)
#         encoded = TimeDistributed(Dense(self.latent_dim, activation='relu'))(flat_inputs)
        
#         # Decoder
#         decoded = TimeDistributed(Dense(np.prod(self.input_shape[1:]), activation='relu'))(encoded)
#         decoded = TimeDistributed(Reshape(self.input_shape[1:]))(decoded)
        
#         autoencoder = Model(inputs, decoded)
#         autoencoder.compile(optimizer=self.optimizer_autoencoder, loss='mse')
        
#         encoder = Model(inputs, encoded)
        
#         return autoencoder, encoder
    
#     def build_rnn_model(self):
#         model = Sequential()
#         model.add(LSTM(self.lstm_units, return_sequences=False, input_shape=(self.input_shape[0], self.latent_dim)))
#         model.add(Dropout(self.dropout_rate))
#         model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
#         model.add(Dropout(self.dropout_rate))
#         model.add(Dense(self.num_classes, activation='softmax'))
#         model.compile(optimizer=self.optimizer_rnn, loss='categorical_crossentropy', metrics=['accuracy'])
#         return model
    
#     def train_autoencoder(self, X_train, X_valid, epochs=40, batch_size=32, verbose=1):
#         self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, X_valid), verbose=verbose)
    
#     def transform(self, X):
#         return self.encoder.predict(X)
    
#     def train_rnn(self, X_train, y_train, X_valid, y_valid, epochs=300, batch_size=32, verbose=1):
#         early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
#         self.history = self.rnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid), callbacks=[early_stopping], verbose=verbose)
    
#     def evaluate_rnn(self, X_test, y_test):
#         return self.rnn_model.evaluate(X_test, y_test)
    
#     def plot_learning_curves(self):
#         history_dict = self.history.history
#         loss = history_dict['loss']
#         val_loss = history_dict['val_loss']
#         accuracy = history_dict['accuracy']
#         val_accuracy = history_dict['val_accuracy']
        
#         epochs = range(1, len(loss) + 1)
        
#         plt.figure(figsize=(14, 5))
        
#         plt.subplot(1, 2, 1)
#         plt.plot(epochs, loss, 'bo', label='Training loss')
#         plt.plot(epochs, val_loss, 'b', label='Validation loss')
#         plt.title('Training and validation loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()
        
#         plt.subplot(1, 2, 2)
#         plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
#         plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
#         plt.title('Training and validation accuracy')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.legend()
        
#         plt.show()