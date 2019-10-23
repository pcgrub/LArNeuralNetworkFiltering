from keras.models import Model
from keras.layers import Dense, LSTM, Input, Reshape

def ilstm_regression(window_size, n_of_neurons):
    inputs1 = Input(shape=(window_size, 1))
    lstm1 = LSTM(n_of_neurons)(inputs1)
    dense1 = Dense(n_of_neurons, activation='relu')(lstm1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    r1 = Reshape((1, 1))(output1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

def lstm_plus_tlfn_regression(window_size, n_of_neurons):
    inputs1 = Input(shape=(window_size, 1))
    lstm1 = LSTM(1, return_sequences=True)(inputs1)
    r1 = Reshape((1, window_size))(lstm1)
    dense1 = Dense(n_of_neurons, activation='relu')(r1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=inputs1, outputs=output1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

def ilstm_classification(window_size, n_of_neurons):
    inputs1 = Input(shape=(window_size, 1))
    lstm1 = LSTM(n_of_neurons)(inputs1)
    dense1 = Dense(n_of_neurons, activation='relu')(lstm1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    r1 = Reshape((1, 1))(output1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def lstm_plus_tlfn_classification(window_size, n_of_neurons):
    inputs1 = Input(shape=(window_size, 1))
    lstm1 = LSTM(1, return_sequences=True)(inputs1)
    r1 = Reshape((1, window_size))(lstm1)
    dense1 = Dense(n_of_neurons, activation='relu')(r1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=inputs1, outputs=output1)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
