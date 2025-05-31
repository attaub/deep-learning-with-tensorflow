import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from tensorflow import keras

from utils import show_reconstructions, rounded_accuracy
from varz import *

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


## Tying weights


class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):

        self.biases = self.add_weight(
            name="bias",
            shape=[self.dense.input_shape[-1]],
            initializer="zeros",
        )

        super().build(batch_input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)


keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

dense_1 = keras.layers.Dense(100, activation="selu")
dense_2 = keras.layers.Dense(30, activation="selu")

# ########################################
# tied_encoder = keras.models.Sequential(
# [keras.layers.Flatten(input_shape=[28, 28]), dense_1, dense_2]
# )

# from keras import Input
# from keras.models import Sequential
# from keras.layers import Flatten

tied_encoder = keras.models.Sequential(
    [keras.Input(shape=(28, 28)), keras.layers.Flatten(), dense_1, dense_2]
)

tied_decoder = keras.models.Sequential(
    [
        DenseTranspose(dense_2, activation="selu"),
        DenseTranspose(dense_1, activation="sigmoid"),
        keras.layers.Reshape([28, 28]),
    ]
)

tied_ae = keras.models.Sequential([tied_encoder, tied_decoder])
# ########################################

tied_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1.5),
    metrics=[rounded_accuracy],
)
history = tied_ae.fit(
    X_train, X_train, epochs=10, validation_data=(X_valid, X_valid)
)

show_reconstructions(tied_ae)
plt.show()

