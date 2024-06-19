from keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

class RNN():
    def __init__(self, X_train, y_train):

        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.5))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(y_train.shape[1], activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
