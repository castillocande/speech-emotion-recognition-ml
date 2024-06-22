from keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import tensorflow as tf

class rnnLSTM():
    def __init__(self, X_train, y_train):
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(64, return_sequences=False))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(y_train.shape[1], activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train(self, X_train, y_train, X_valid, y_valid, epochs=10, batch_size=32, random_seed=42):
        tf.random.set_seed(random_seed)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid))

    def evaluate(self, X_test, y_test):
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        return test_loss, test_accuracy


