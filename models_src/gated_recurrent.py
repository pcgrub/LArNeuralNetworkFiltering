"""
Neural networks including GRU-type neurons
author: Clemens Grubitz (mailto:clemens@grubitz.eu)
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Input, Reshape


def gru_only(window_size):
    inputs1 = Input(shape=(window_size, 1))
    gru1 = GRU(1)(inputs1)
    r1 = Reshape((1, 1))(gru1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

def gru_only_stacked(window_size, number_of_layers):
    inputs1 = Input(shape=(window_size, 1))
    hidden_layers = [inputs1]
    for i in range(number_of_layers-1):
        hidden_layers.append(GRU(1,return_sequences=True)
                             (hidden_layers[i]))
    hidden_layers.append(GRU(1)(hidden_layers[-1]))
    r1 = Reshape((1, 1))(hidden_layers[-1])
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

def igru_regression(window_size, n_of_neurons):
    inputs1 = Input(shape=(window_size, 1))
    gru1 = GRU(n_of_neurons)(inputs1)
    dense1 = Dense(n_of_neurons, activation='relu')(gru1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    r1 = Reshape((1, 1))(output1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

def gru_plus_tlfn_regression(window_size, n_of_neurons):
    inputs1 = Input(shape=(window_size, 1))
    gru1 = GRU(1, return_sequences=True)(inputs1)
    r1 = Reshape((1, window_size))(gru1)
    dense1 = Dense(n_of_neurons, activation='relu')(r1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=inputs1, outputs=output1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

def igru_classification(window_size, n_of_neurons):
    inputs1 = Input(shape=(window_size, 1))
    gru1 = GRU(n_of_neurons)(inputs1)
    dense1 = Dense(n_of_neurons, activation='relu')(gru1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    r1 = Reshape((1, 1))(output1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def gru_plus_tlfn_classification(window_size, n_of_neurons):
    inputs1 = Input(shape=(window_size, 1))
    gru1 = GRU(1, return_sequences=True)(inputs1)
    r1 = Reshape((1, window_size))(gru1)
    dense1 = Dense(n_of_neurons, activation='relu')(r1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=inputs1, outputs=output1)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

