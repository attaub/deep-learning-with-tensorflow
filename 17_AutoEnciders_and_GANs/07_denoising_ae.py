import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
import tensorflow as tf
from tensorflow import keras

#################################################################
## Stacked denoising Autoencoder
#################################################################
# Using Gaussian noise:

denoising_encoder = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.GaussianNoise(0.2),
        keras.layers.Dense(100, activation="selu"),
        keras.layers.Dense(30, activation="selu"),
    ]
)
denoising_decoder = keras.models.Sequential(
    [
        keras.layers.Dense(100, activation="selu", input_shape=[30]),
        keras.layers.Dense(28 * 28, activation="sigmoid"),
        keras.layers.Reshape([28, 28]),
    ]
)
denoising_ae = keras.models.Sequential([denoising_encoder, denoising_decoder])

denoising_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1.0),
    metrics=[rounded_accuracy],
)
history = denoising_ae.fit(
    X_train, X_train, epochs=10, validation_data=(X_valid, X_valid)
)


tf.random.set_seed(42)
np.random.seed(42)

noise = keras.layers.GaussianNoise(0.2)
show_reconstructions(denoising_ae, noise(X_valid, training=True))
plt.show()

