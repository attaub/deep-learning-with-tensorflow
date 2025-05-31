import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
import tensorflow as tf
from tensorflow import keras
from utils import rounded_accuracy, show_reconstructions
from varz import *

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)


## Using Convolutional Layers Instead of Dense Layers

tf.random.set_seed(42)
np.random.seed(42)

conv_encoder = keras.models.Sequential(
    [
        keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
        keras.layers.Conv2D(16, kernel_size=3, padding="SAME", activation="selu"),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="selu"),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="selu"),
        keras.layers.MaxPool2D(pool_size=2),
    ]
)

conv_decoder = keras.models.Sequential(
    [
        keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="VALID", activation="selu", input_shape=[3, 3, 64],),
        keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="SAME", activation="selu"),
        keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="SAME", activation="sigmoid"),
        keras.layers.Reshape([28, 28]),
    ]
)

conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

conv_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1.0),
    metrics=[rounded_accuracy],
)

history = conv_ae.fit(
    X_train, X_train, epochs=5, validation_data=(X_valid, X_valid)
)

conv_encoder.summary()
conv_decoder.summary()
conv_ae.summary()

# def show_reconstructions(model, images=X_valid, n_images=5):
show_reconstructions(conv_ae, X_valid)
plt.show()
