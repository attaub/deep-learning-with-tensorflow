import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
import tensorflow as tf
from tensorflow import keras

from utils import *

# # Sparse Autoencoder
# build a simple stacked autoencoder compare it to the sparse autoencoders
# use sigmoid activation for coding layer, to ensure the coding values range [0, 1]

simple_encoder = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(100, activation="selu"),
        keras.layers.Dense(30, activation="sigmoid"),
    ]
)

simple_decoder = keras.models.Sequential(
    [
        keras.layers.Dense(100, activation="selu", input_shape=[30]),
        keras.layers.Dense(28 * 28, activation="sigmoid"),
        keras.layers.Reshape([28, 28]),
    ]
)

simple_ae = keras.models.Sequential([simple_encoder, simple_decoder])

simple_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1.0),
    metrics=[rounded_accuracy],
)

history = simple_ae.fit(
    X_train, X_train, epochs=10, validation_data=(X_valid, X_valid)
)

show_reconstructions(simple_ae)
plt.show()

#################################################################

plot_activations_histogram(simple_encoder, height=0.35)
plt.show()


# Now let's add l_1 regularization to the coding layer:

tf.random.set_seed(42)
np.random.seed(42)

sparse_l1_encoder = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(100, activation="selu"),
        keras.layers.Dense(300, activation="sigmoid"),
        keras.layers.ActivityRegularization(
            l1=1e-3
            # or activity_regularizer=keras.regularizers.l1(1e-3) to previous layer.
            # keras.layers.Dense(300, activation="sigmoid", activity_regularizer= keras.regularizers.l1(1e-3)),
        ),
    ]
)

sparse_l1_decoder = keras.models.Sequential(
    [
        keras.layers.Dense(100, activation="selu", input_shape=[300]),
        keras.layers.Dense(28 * 28, activation="sigmoid"),
        keras.layers.Reshape([28, 28]),
    ]
)

sparse_l1_ae = keras.models.Sequential([sparse_l1_encoder, sparse_l1_decoder])

sparse_l1_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1.0),
    metrics=[rounded_accuracy],
)

history = sparse_l1_ae.fit(
    X_train, X_train, epochs=10, validation_data=(X_valid, X_valid)
)

show_reconstructions(sparse_l1_ae)

plot_activations_histogram(sparse_l1_encoder, height=1.0)
plt.show()

# Let's use the KL Divergence loss instead to ensure sparsity, and target 10% sparsity rather than 0%:

p = 0.1
q = np.linspace(0.001, 0.999, 500)
kl_div = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
mse = (p - q) ** 2
mae = np.abs(p - q)

plt.plot([p, p], [0, 0.3], "k:")
plt.text(0.05, 0.32, "Target\nsparsity", fontsize=14)
plt.plot(q, kl_div, "b-", label="KL divergence")
plt.plot(q, mae, "g--", label=r"MAE ($\ell_1$)")
plt.plot(q, mse, "r--", linewidth=1, label=r"MSE ($\ell_2$)")
plt.legend(loc="upper left", fontsize=14)
plt.xlabel("Actual sparsity")
plt.ylabel("Cost", rotation=0)
plt.axis([0, 1, 0, 0.95])
save_fig("sparsity_loss_plot")

K = keras.backend

kl_divergence = keras.losses.kullback_leibler_divergence


class KLDivergenceRegularizer(keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1):
        self.weight = weight
        self.target = target

    def __call__(self, inputs):
        mean_activities = K.mean(inputs, axis=0)
        return self.weight * (
            kl_divergence(self.target, mean_activities)
            + kl_divergence(1.0 - self.target, 1.0 - mean_activities)
        )


tf.random.set_seed(42)
np.random.seed(42)

kld_reg = KLDivergenceRegularizer(weight=0.05, target=0.1)
sparse_kl_encoder = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(100, activation="selu"),
        keras.layers.Dense(
            300, activation="sigmoid", activity_regularizer=kld_reg
        ),
    ]
)

sparse_kl_decoder = keras.models.Sequential(
    [
        keras.layers.Dense(100, activation="selu", input_shape=[300]),
        keras.layers.Dense(28 * 28, activation="sigmoid"),
        keras.layers.Reshape([28, 28]),
    ]
)

sparse_kl_ae = keras.models.Sequential([sparse_kl_encoder, sparse_kl_decoder])

sparse_kl_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1.0),
    metrics=[rounded_accuracy],
)
history = sparse_kl_ae.fit(
    X_train, X_train, epochs=10, validation_data=(X_valid, X_valid)
)

show_reconstructions(sparse_kl_ae)

plot_activations_histogram(sparse_kl_encoder)
save_fig("sparse_autoencoder_plot")
plt.show()
