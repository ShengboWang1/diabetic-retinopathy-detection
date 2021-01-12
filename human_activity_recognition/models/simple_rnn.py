import numpy as np
import tensorflow as tf
import gin
from tensorflow.keras import layers, Sequential


@gin.configurable
def simple_rnn(n_neurons):

    model = Sequential()
    model.add(layers.LSTM(n_neurons, input_shape=(250, 6), return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(13))

    return model
