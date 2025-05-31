import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
import tensorflow as tf
from tensorflow import keras
from varz import *


def train_autoencoder(
    n_neurons,
    X_train,
    X_valid,
    loss,
    optimizer,
    n_epochs=10,
    output_activation=None,
    metrics=None,
):
    n_inputs = X_train.shape[-1]
    encoder = keras.models.Sequential(
        [
            keras.layers.Dense(
                n_neurons, activation="selu", input_shape=[n_inputs]
            )
        ]
    )
    decoder = keras.models.Sequential(
        [
            keras.layers.Dense(n_inputs, activation=output_activation),
        ]
    )
    autoencoder = keras.models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer, loss, metrics=metrics)
    autoencoder.fit(
        X_train, X_train, epochs=n_epochs, validation_data=(X_valid, X_valid)
    )
    return encoder, decoder, encoder(X_train), encoder(X_valid)


K = keras.backend

X_train_flat = K.batch_flatten(X_train)  # equivalent to .reshape(-1, 28 * 28)
X_valid_flat = K.batch_flatten(X_valid)

enc1, dec1, X_train_enc1, X_valid_enc1 = train_autoencoder(
    100,
    X_train_flat,
    X_valid_flat,
    "binary_crossentropy",
    keras.optimizers.SGD(learning_rate=1.5),
    output_activation="sigmoid",
    metrics=[rounded_accuracy],
)

enc2, dec2, _, _ = train_autoencoder(
    30,
    X_train_enc1,
    X_valid_enc1,
    "mse",
    keras.optimizers.SGD(learning_rate=0.05),
    output_activation="selu",
)

stacked_ae_1_by_1 = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        enc1,
        enc2,
        dec2,
        dec1,
        keras.layers.Reshape([28, 28]),
    ]
)

# plt.show() # ???

stacked_ae_1_by_1.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=0.1),
    metrics=[rounded_accuracy],
)
history = stacked_ae_1_by_1.fit(
    X_train, X_train, epochs=10, validation_data=(X_valid, X_valid)
)

show_reconstructions(stacked_ae_1_by_1)
plt.show()
