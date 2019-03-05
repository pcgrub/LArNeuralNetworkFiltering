from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, Input, Reshape
from .loss_functions import new_metrics


def OptimalFilter(dim):
    ini = 'RandomUniform'
    opti = SGD(lr=0.1)
    inputs = Input(shape=(dim, 1))
    r1 = Reshape((1, dim))(inputs)
    dense1 = Dense(1, activation='sigmoid', kernel_initializer=ini)(r1)
    model = Model(inputs=inputs, outputs=dense1)

    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=[new_metrics])
    print(model.summary())
    return model
