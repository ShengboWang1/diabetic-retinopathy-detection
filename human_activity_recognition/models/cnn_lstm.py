import gin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D


@gin.configurable
def cnn_lstm(n_filters, dropout_rate, window_size, lstm_units, dense_units):
    # define model
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(n_filters, kernel_size=3, activation='relu'), input_shape=(None, 250, 6)))
    model.add(TimeDistributed(Conv1D(n_filters, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(dropout_rate)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(lstm_units, input_shape=(window_size, 6), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(12, activation='softmax'))

    return model
