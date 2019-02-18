from keras.models import Model
from keras.layers import Dense, Input, Reshape
from .loss_functions import *


def BaselineReLU(dim, n):
    ini = 'RandomUniform'
    inputs1 = Input(shape=(dim, 1))
    r1 = Reshape((1, dim))(inputs1)
    dense1 = Dense(dim, activation='relu', kernel_initializer=ini)(r1)
    dense2 = Dense(dim, activation='relu', kernel_initializer=ini)(dense1)
    dense = Dense(1, activation='relu', kernel_initializer=ini)(dense2)
    model = Model(inputs=inputs1, outputs=dense)
    model.compile(loss='mean_squared_error', optimizer='RMSprop')
    print(model.summary())
    return model


def Baseline(dim, n):
    ini = 'RandomUniform'
    inputs1 = Input(shape=(dim, 1))
    r1 = Reshape((1, dim))(inputs1)
    dense1 = Dense(dim, activation='relu', kernel_initializer=ini)(r1)
    dense2 = Dense(dim, activation='relu', kernel_initializer=ini)(dense1)
    dense = Dense(1, kernel_initializer=ini)(dense2)
    model = Model(inputs=inputs1, outputs=dense)
    model.compile(loss='mean_squared_error', optimizer='RMSprop')
    print(model.summary())
    return model


def TLFN(dim, n):
    ini = 'RandomUniform'
    inputs1 = Input(shape=(dim, 1))
    r1 = Reshape((1, dim))(inputs1)
    dense1 = Dense(dim, activation='relu', kernel_initializer=ini)(r1)
    dense = Dense(1, kernel_initializer=ini)(dense1)
    model = Model(inputs=inputs1, outputs=dense)
    model.compile(loss='mean_squared_error', optimizer='RMSprop')
    print(model.summary())
    return model


def TLFNNoReLU(dim, n):
    ini = 'RandomUniform'
    inputs1 = Input(shape=(dim, 1))
    r1 = Reshape((1, dim))(inputs1)
    dense1 = Dense(dim, kernel_initializer=ini)(r1)
    dense = Dense(1, kernel_initializer=ini)(dense1)
    model = Model(inputs=inputs1, outputs=dense)
    model.compile(loss='mean_squared_error', optimizer='RMSprop')
    print(model.summary())
    return model
