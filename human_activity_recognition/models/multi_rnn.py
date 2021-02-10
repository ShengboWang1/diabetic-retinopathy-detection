import gin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU


@gin.configurable
def multi_rnn(n_lstm, n_dense, dropout_rate, window_size, rnn_units, dense_units, rnn_type, kernel_initializer):
    model = Sequential()
    for i in range(1, n_lstm):
        if rnn_type == 'LSTM':
            model.add(LSTM(rnn_units, input_shape=(window_size, 6), return_sequences=True, kernel_initializer=kernel_initializer))
        elif rnn_type == 'simeple_rnn':
            model.add(SimpleRNN(rnn_units, input_shape=(window_size, 6), return_sequences=True, kernel_initializer=kernel_initializer))
        elif rnn_type == 'GRU':
            model.add(GRU(rnn_units, input_shape=(window_size, 6), return_sequences=True, kernel_initializer=kernel_initializer))
        else:
            return ValueError

    # model.add(layers.LSTM(128, input_shape=(window_size, 6), return_sequences=True))
    # model.add(LSTM(64, input_shape=(window_size, 6), return_sequences=True))
    # model.add(layers.Bidirectional(layers.LSTM(n_neurons, input_shape=(window_size, 6), return_sequences=True)))
    # model.add(layers.Bidirectional(layers.LSTM(32, input_shape=(n_neurons, 6), return_sequences=True)))
    model.add(Dropout(dropout_rate))
    for i in range(1, n_dense):
        model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(12, activation='softmax'))

    return model
