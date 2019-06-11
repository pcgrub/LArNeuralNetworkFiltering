from keras.models import Sequential, Model
from keras.layers import GRU, Dense, LSTM, Input, Reshape
from keras.regularizers import l2
from .loss_functions import *


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

def GatedRecurrent9GRUClassification(dim, l):
    # l=9 for Signal
    # 10 GRUS, 5 Dense for Pile Up
    inputs1 = Input(shape=(dim, 1))
    gru1 = GRU(l)(inputs1)
    dense1 = Dense(l, activation='relu')(gru1)
    output1 = Dense(1)(dense1)
    r1 = Reshape((1, 1))(output1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    print(model.summary())
    return model


def GatedRecurrentTLFNClassification(dim, l):
    # l=12 for Signal
    # (10,0,10) for Pile Up
    inputs1 = Input(shape=(dim, 1))
    gru1 = GRU(1, return_sequences=True)(inputs1)
    r1 = Reshape((1, dim))(gru1)
    dense1 = Dense(l, activation='relu')(r1)
    output1 = Dense(1)(dense1, activation='sigmoid')
    model = Model(inputs=inputs1, outputs=output1)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    print(model.summary())
    return model

def GatedRecurrentNonSequentialStacked(dim,n,l):
    inputs1 = Input(shape=(dim, 1))
    hidden_layers = []
    hidden_layers.append(GRU(l)(inputs1))
    for i in range(n):
        hidden_layers.append(Dense(l, activation='relu')(hidden_layers[i]))
    output1 = Dense(1)(hidden_layers[-1])
    r1 = Reshape((1, 1))(output1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model


def GatedRecurrentTLFNStacked(dim, n, l):
    # l=12 for Signal
    # (10,0,10) for Pile Up
    inputs1 = Input(shape=(dim, 1))
    gru1 = GRU(1, return_sequences=True)(inputs1)
    hidden_layers = []
    hidden_layers.append(Reshape((1, dim))(gru1))
    for i in range(n):
        hidden_layers.append(Dense(l)(hidden_layers[i]))
    output1 = Dense(1)(hidden_layers[-1])
    model = Model(inputs=inputs1, outputs=output1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model


def GatedRecurrentTLFNreg(dim, l):
    # l=12 for Signal
    # (10,0,10) for Pile Up
    inputs1 = Input(shape=(dim, 1))
    gru1 = GRU(1, return_sequences=True)(inputs1)
    r1 = Reshape((1, dim))(gru1)
    dense1 = Dense(l, activation='relu', kernel_regularizer=l2(0.01))(r1)
    output1 = Dense(1, kernel_regularizer=l2(0.01))(dense1)
    model = Model(inputs=inputs1, outputs=output1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model


def GatedRecurrentNonSequentialreg(dim, l):
    # l=9 for Signal
    # 10 GRUS, 5 Dense for Pile Up
    inputs1 = Input(shape=(dim, 1))
    gru1 = GRU(l)(inputs1)
    dense1 = Dense(l, activation='relu', kernel_regularizer=l2(0.01))(gru1)
    output1 = Dense(1, kernel_regularizer=l2(0.01))(dense1)
    r1 = Reshape((1, 1))(output1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

