import gin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


@gin.configurable
def multi_lstm(n_lstm, n_dense, dropout_rate, window_size, lstm_units, dense_units):
    model = Sequential()
    for i in range(1, n_lstm):
        model.add(LSTM(lstm_units, input_shape=(window_size, 6), return_sequences=True))
    # model.add(layers.LSTM(128, input_shape=(window_size, 6), return_sequences=True))
    # model.add(LSTM(64, input_shape=(window_size, 6), return_sequences=True))
    # model.add(layers.Bidirectional(layers.LSTM(n_neurons, input_shape=(window_size, 6), return_sequences=True)))
    # model.add(layers.Bidirectional(layers.LSTM(32, input_shape=(n_neurons, 6), return_sequences=True)))
    model.add(Dropout(dropout_rate))
    for i in range(1, n_dense):
        model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(12, activation='softmax'))

    return model


