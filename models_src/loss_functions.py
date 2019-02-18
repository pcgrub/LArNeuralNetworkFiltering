from keras import backend as K
import tensorflow as tf


def weighted_loss(y_true, y_pred):

    # numerator according to normal mean_squared_error
    se = K.square(y_pred - y_true)

    # denominator taking into account the detector resolution
    # resulting in \sqrt(E_{act})

    # in case of actual energy $E_{act}=0$, the denomiator becomes
    # $\sqrt{E_{pred}}$ instead.
    truth = K.not_equal(y_true, K.zeros_like(y_true))
    d = K.switch(truth, y_true, y_pred)

    # in case of both being zero the prediction is correct.
    # all resulting 'nan's will be replaced by an error of zero

    has_nans = se / K.abs(d)
    weighted_error = tf.where(tf.is_nan(has_nans), tf.zeros_like(has_nans),
                              has_nans)
    return K.mean(weighted_error, axis=-1)


def weighted_loss2(y_true, y_pred):

    # numerator according to normal mean_squared_error
    offset = 100
    se = K.abs(y_pred - y_true)

    d = K.sqrt(K.abs(y_true)) + K.ones_like(y_true) * offset
    weighted_error = se / d
    return K.mean(weighted_error, axis=-1)


def new_metrics(y_true, y_pred):
    return weighted_loss(y_true, y_pred)
