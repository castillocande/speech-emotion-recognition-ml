from keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt

        
class rnnLSTM():
    """Clase para definir, entrenar y evaluar el modelo RNN LSTM."""
    def __init__(self, X_train, y_train, lr = 0.001, patience = 3, dropout_rate = 0.5, momentum = 0.9, seed = 42):
        self.seed = seed
        tf.random.set_seed(self.seed)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(y_train.shape[1], activation='softmax'))
        self.model.compile(optimizer=SGD(learning_rate=lr, momentum=momentum), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_valid, y_valid, epochs=10, batch_size=32):
        """Entrena el modelo RNN LSTM."""
        tf.random.set_seed(self.seed)
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid), callbacks=[self.early_stopping])

    def evaluate(self, X_test, y_test):
        """Evalúa el modelo RNN LSTM."""
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        return test_loss, test_accuracy
    
    def plot_learning_curves(self):
        """Grafica las curvas de aprendizaje."""
        history_dict = self.history.history
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        accuracy = history_dict['accuracy']
        val_accuracy = history_dict['val_accuracy']
        
        epochs = range(1, len(loss) + 1)
        
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'g', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, 'g', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.show()