import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


#################################################################
def logit(z):
    return 1 / (1 + np.exp(-z))


z = np.linspace(-5, 5, 200)

# Leaky ReLU


def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha * z, z)

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
# Now let's try PReLU:


tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, kernel_initializer="he_normal"),
        keras.layers.PReLU(),
        keras.layers.Dense(100, kernel_initializer="he_normal"),
        keras.layers.PReLU(),
        keras.layers.Dense(10, activation="softmax"),
    ]
)


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1e-3),
    metrics=["accuracy"],
)


history = model.fit(
    X_train, y_train, epochs=10, validation_data=(X_valid, y_valid)
)

