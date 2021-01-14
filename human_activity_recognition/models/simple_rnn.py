from tensorflow.keras.models import Sequential
import gin
from tensorflow.keras.layers import Dense, Dropout, LSTM


@gin.configurable
def simple_rnn(n_neurons, dense_units, dropout_rate, window_size):

    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(window_size, 6), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(13, activation='softmax'))
    return model
