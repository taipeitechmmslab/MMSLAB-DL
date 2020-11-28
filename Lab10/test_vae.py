import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


size = 28
n = 15
save_images = np.zeros((size * n, size * n, 1))
# linearly spaced coordinates corresponding to the 2D plot of digit classes in the latent space
grid_x = np.linspace(-1.5, 1.5, n)
grid_y = np.linspace(-1.5, 1.5, n)
model = tf.keras.models.load_model('logs_vae/models/best_model.h5')
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        img = model(z_sample)
        save_images[i * size: (i + 1) * size, j * size: (j + 1) * size] = img.numpy()[0]

plt.imshow(save_images[..., 0], cmap='gray')
plt.show()
plt.imsave('output.png', save_images[..., 0], cmap='gray')