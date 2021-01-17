from tensorflow.keras.models import Sequential
import gin
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN


@gin.configurable
def simple_rnn(rnn_type, n_neurons, dense_units, dropout_rate, window_size):

    model = Sequential(name='simple_rnn')
    if rnn_type == 'simple_rnn':
        model.add(SimpleRNN(n_neurons, input_shape=(window_size, 6), return_sequences=True))
    elif rnn_type == 'LSTM':
        model.add(LSTM(n_neurons, input_shape=(window_size, 6), return_sequences=True))
    elif rnn_type == 'GRU':
        model.add(GRU(n_neurons, input_shape=(window_size, 6), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    return model
