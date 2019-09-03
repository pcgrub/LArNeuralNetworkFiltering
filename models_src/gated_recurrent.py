from keras.models import Sequential, Model
from keras.layers import GRU, Dense, LSTM, Input, Reshape
from keras.regularizers import l2
from .loss_functions import *

def GruOnly(dim):
    inputs1 = Input(shape=(dim, 1))
    gru1 = GRU(1)(inputs1)
    r1 = Reshape((1, 1))(gru1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

def GruOnlyStacked(dim, n):
    inputs1 = Input(shape=(dim, 1))
    hidden_layers = [inputs1]
    for i in range(n-1):
        hidden_layers.append(GRU(1, return_sequences=True)(hidden_layers[i]))
    hidden_layers.append(GRU(1)(hidden_layers[-1]))
    r1 = Reshape((1, 1))(hidden_layers[-1])
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='mean_squared_error', optimizer='adam')
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

def GatedRecurrent9GRUClassification(dim, l):
    # l=9 for Signal
    # 10 GRUS, 5 Dense for Pile Up
    inputs1 = Input(shape=(dim, 1))
    gru1 = GRU(l)(inputs1)
    dense1 = Dense(l, activation='tanh')(gru1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    r1 = Reshape((1, 1))(output1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def GatedRecurrentTLFNClassification(dim, l):
    # l=12 for Signal
    # (10,0,10) for Pile Up
    inputs1 = Input(shape=(dim, 1))
    gru1 = GRU(1, return_sequences=True)(inputs1)
    r1 = Reshape((1, dim))(gru1)
    dense1 = Dense(l, activation='sigmoid')(r1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=inputs1, outputs=output1)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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

def LSTMNonSequential(dim, l):
    # l=9 for Signal
    # 10 LSTMS, 5 Dense for Pile Up
    inputs1 = Input(shape=(dim, 1))
    lstm1 = LSTM(l)(inputs1)
    dense1 = Dense(l, activation='relu')(lstm1)
    output1 = Dense(1)(dense1)
    r1 = Reshape((1, 1))(output1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

def LSTMTLFN(dim, l):
    # l=12 for Signal
    # (10,0,10) for Pile Up
    inputs1 = Input(shape=(dim, 1))
    lstm1 = LSTM(1, return_sequences=True)(inputs1)
    r1 = Reshape((1, dim))(lstm1)
    dense1 = Dense(l, activation='relu')(r1)
    output1 = Dense(1)(dense1)
    model = Model(inputs=inputs1, outputs=output1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

def LSTM9Classification(dim, l):
    # l=9 for Signal
    # 10 LSTMS, 5 Dense for Pile Up
    inputs1 = Input(shape=(dim, 1))
    lstm1 = LSTM(l)(inputs1)
    dense1 = Dense(l, activation='tanh')(lstm1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    r1 = Reshape((1, 1))(output1)
    model = Model(inputs=inputs1, outputs=r1)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def LSTMTLFNClassification(dim, l):
    # l=12 for Signal
    # (10,0,10) for Pile Up
    inputs1 = Input(shape=(dim, 1))
    lstm1 = LSTM(1, return_sequences=True)(inputs1)
    r1 = Reshape((1, dim))(lstm1)
    dense1 = Dense(l, activation='sigmoid')(r1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=inputs1, outputs=output1)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

