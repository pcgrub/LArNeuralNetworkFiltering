from keras.models import Sequential, Model
from keras.layers import GRU, Dense, LSTM, Input, Reshape
from .loss_functions import *


def GatedRecurrent(dim, n, l):
    model = Sequential()
    for i in range(n - 1):
        model.add(LSTM(l, input_shape=(1, dim), return_sequences=True))
    model.add(GRU(l, return_sequences=True))
    # model.add(Dense(l, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='RMSprop')
    print(model.summary())
    return model


def GatedRecurrentNonSequential(dim, l):
    # l=9 for Signal
    # 10 GRUS, 5 Dense for Pile Up
    inputs1 = Input(shape=(dim, 1))
    gru1 = GRU(l)(inputs1)
    dense1 = Dense(l, activation='relu')(gru1)
    output1 = Dense(1)(dense1)
    r1 = Reshape((1, 1))(output1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model


def GatedRecurrentTLFN(dim, l):
    # l=12 for Signal
    # (10,0,10) for Pile Up
    inputs1 = Input(shape=(dim, 1))
    gru1 = GRU(1, return_sequences=True)(inputs1)
    r1 = Reshape((1, dim))(gru1)
    dense1 = Dense(l, activation='relu')(r1)
    output1 = Dense(1)(dense1)
    model = Model(inputs=inputs1, outputs=output1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model


def GatedRecurrentNonSequentialAllReLU(dim, n, l):
    inputs1 = Input(shape=(dim, 1))
    gru1 = GRU(l, activation='relu')(inputs1)
    dense1 = Dense(l, activation='relu')(gru1)
    output1 = Dense(1)(dense1)
    r1 = Reshape((1, 1))(output1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='mean_squared_error', optimizer='RMSprop')
    print(model.summary())
    return model


def GatedRecurrentTLFNAllReLU(dim, n, l):
    inputs1 = Input(shape=(dim, 1))
    gru1 = GRU(1, activation='relu', return_sequences=True)(inputs1)
    r1 = Reshape((1, dim))(gru1)
    dense1 = Dense(dim, activation='relu')(r1)
    output1 = Dense(1)(dense1)
    model = Model(inputs=inputs1, outputs=output1)
    model.compile(loss='mean_squared_error', optimizer='RMSprop')
    print(model.summary())
    return model


def GatedRecurrentNewLoss(dim, n, l):
    inputs1 = Input(shape=(dim, 1))
    gru1 = GRU(1, return_sequences=True)(inputs1)
    r1 = Reshape((1, dim))(gru1)
    dense1 = Dense(l, activation='relu')(r1)
    output1 = Dense(1)(dense1)
    model = Model(inputs=inputs1, outputs=output1)
    model.compile(loss=weighted_loss, optimizer='RMSprop')
    print(model.summary())
    return model

