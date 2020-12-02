import tensorflow as tf


def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(y_true, y_pred), axis=[1, 2, 3]))


def reconstruction_loss(y_true, y_pred):
    bce = y_true * tf.math.log(y_pred + 1e-07) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-07)
    return tf.reduce_mean(tf.reduce_sum(-bce, axis=[1, 2, 3]))
