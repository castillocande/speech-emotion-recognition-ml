from keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

class rnnLSTM():
    def __init__(self, X_train, y_train, lr = 0.001, patience = 3, dropout_rate = 0.5):
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(128, return_sequences=False))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(y_train.shape[1], activation='softmax'))
        self.model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_valid, y_valid, epochs=10, batch_size=32, random_seed=42):
        print("cambio")
        tf.random.set_seed(random_seed)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid), callbacks=[self.early_stopping])

    def evaluate(self, X_test, y_test):
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        return test_loss, test_accuracy


