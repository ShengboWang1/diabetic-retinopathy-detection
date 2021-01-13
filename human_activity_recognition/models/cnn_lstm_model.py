import tensorflow as tf
import gin
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

@gin.configurable
def cnn_lstm_model(window_size):
    model = Sequential()
    model.add(TimeDistributed(
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        input_shape=(None, window_size, 6)))
    model.add(TimeDistributed(
        Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(13, activation='softmax'))

    return model