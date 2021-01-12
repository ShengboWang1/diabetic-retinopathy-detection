import numpy as np
import tensorflow as tf
import gin
from tensorflow.keras import layers, Sequential


@gin.configurable
def simple_rnn(n_neurons, dense_units, dropout_rate, window_size, rnn_units=30):

    model = Sequential()
    model.add(layers.LSTM(256, input_shape=(window_size, 6), return_sequences=True))
    # model.add(layers.LSTM(128, input_shape=(window_size, 6), return_sequences=True))
    # model.add(layers.LSTM(64, input_shape=(window_size, 6), return_sequences=True))
    # model.add(layers.Bidirectional(layers.LSTM(n_neurons, input_shape=(window_size, 6), return_sequences=True)))
    # model.add(layers.Bidirectional(layers.LSTM(32, input_shape=(n_neurons, 6), return_sequences=True)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dense(13))

    return model
