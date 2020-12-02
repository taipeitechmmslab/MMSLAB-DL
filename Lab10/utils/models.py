import numpy as np
import tensorflow as tf
from tensorflow import keras


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(z_var) * epsilon


def create_vae_model(input_shape, latent_dim):
    # Define encoder model.
    img_inputs = keras.Input(input_shape)
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(img_inputs)
    x = keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    shape_before_flatten = x.shape
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(32, 'relu')(x)
    z_mean = keras.layers.Dense(latent_dim)(x)
    z_var = keras.layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_var])
    encoder = keras.Model(inputs=img_inputs, outputs=z, name='encoder')
    encoder.summary()

    # Define decoder model.
    latent_inputs = keras.Input((latent_dim,))
    x = keras.layers.Dense(np.prod(shape_before_flatten[1:]), activation='relu')(latent_inputs)
    x = keras.layers.Reshape(target_shape=shape_before_flatten[1:])(x)
    x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    img_outputs = keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    decoder = keras.Model(inputs=latent_inputs, outputs=img_outputs, name='decoder')
    decoder.summary()

    # Define VAE model.
    z = encoder(img_inputs)
    img_outputs = decoder(z)
    vae = keras.Model(inputs=img_inputs, outputs=img_outputs, name='vae')

    # add KL loss
    kl_loss = 0.5 * tf.reduce_mean(tf.square(z_mean) - 1 - z_var + tf.exp(z_var))

    vae.add_loss(kl_loss)
    return vae