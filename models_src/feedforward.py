"""
Time-lagged feedforward network with one hidden layer
author: Clemens Grubitz (mailto:clemens@grubitz.eu)
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Reshape

def tlfn(window_size, n_of_neurons):
    ini = 'RandomUniform'
    inputs1 = Input(shape=(window_size, 1))
    r1 = Reshape((1, window_size))(inputs1)
    dense1 = Dense(n_of_neurons, activation='relu',
                   kernel_initializer=ini)(r1)
    dense = Dense(1, activation='sigmoid',
                  kernel_initializer=ini)(dense1)
    model = Model(inputs=inputs1, outputs=dense)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model
