import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from functools import partial

#################################################################

(X_train_full, y_train_full), (
    X_test,
    y_test,
) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

#################################################################

pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds


#################################################################
# # Avoiding Overfitting Through Regularization

# ## $\ell_1$ and $\ell_2$ regularization


layer = keras.layers.Dense(
    100,
    activation="elu",
    kernel_initializer="he_normal",
    kernel_regularizer=keras.regularizers.l2(0.01),
)
# or l1(0.1) for ℓ1 regularization with a factor of 0.1
# or l1_l2(0.1, 0.01) for both ℓ1 and ℓ2 regularization, with factors 0.1 and 0.01 respectively

#################################################################

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(
            300,
            activation="elu",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(0.01),
        ),
        keras.layers.Dense(
            100,
            activation="elu",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(0.01),
        ),
        keras.layers.Dense(
            10,
            activation="softmax",
            kernel_regularizer=keras.regularizers.l2(0.01),
        ),
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="nadam",
    metrics=["accuracy"],
)

n_epochs = 2

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=n_epochs,
    validation_data=(X_valid_scaled, y_valid),
)



RegularizedDense = partial(
    keras.layers.Dense,
    activation="elu",
    kernel_initializer="he_normal",
    kernel_regularizer=keras.regularizers.l2(0.01),
)

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        RegularizedDense(300),
        RegularizedDense(100),
        RegularizedDense(10, activation="softmax"),
    ]
)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="nadam",
    metrics=["accuracy"],
)
n_epochs = 2
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=n_epochs,
    validation_data=(X_valid_scaled, y_valid),
)
