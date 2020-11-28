# import packages
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from utils.models import create_vae_model
from utils.losses import reconstruction_loss
from utils.callbacks import SaveDecoderOutput, SaveDecoderModel
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_fn(dataset, input_size=(28, 28)):
    x = tf.cast(dataset['image'], tf.float32)
    x = tf.image.resize(x, input_size)
    x = x / 255.
    return x, x


dataset = 'mnist'     # 'cifar10', 'fashion_mnist', 'mnist'
log_dirs = 'logs_vae'
batch_size = 16
latent_dim = 2
input_shape = (28, 28, 1)

# Load datasets
train_data = tfds.load(dataset, split=tfds.Split.TRAIN)
test_data = tfds.load(dataset, split=tfds.Split.TEST)

# Setting datasets
AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
train_data = train_data.shuffle(1000)
train_data = train_data.map(parse_fn, num_parallel_calls=AUTOTUNE)
train_data = train_data.batch(batch_size)
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
test_data = test_data.map(parse_fn, num_parallel_calls=AUTOTUNE)
test_data = test_data.batch(batch_size)
test_data = test_data.prefetch(buffer_size=AUTOTUNE)

# Callbacks function
model_dir = log_dirs + '/models'
os.makedirs(model_dir, exist_ok=True)
model_tb = keras.callbacks.TensorBoard(log_dir=log_dirs)
model_sdw = SaveDecoderModel(model_dir + '/best_model.h5', monitor='val_loss')
model_testd = SaveDecoderOutput(28, log_dir=log_dirs)

# create vae model
vae_model = create_vae_model(input_shape, latent_dim)

# training
optimizer = tf.keras.optimizers.RMSprop()
vae_model.compile(optimizer, loss=reconstruction_loss)
vae_model.fit(train_data, epochs=20, validation_data=test_data, callbacks=[model_tb, model_sdw, model_testd])
